"""Memory Service for managing claims, storage, extraction, and conflict resolution."""

from datetime import datetime, timezone
import json
import re
from typing import Any, Dict, List, Optional, Tuple
from pydantic import ValidationError

from core.conflict_detector import evaluate_claim_conflict
from core.database import DatabaseManager, DEFAULT_DB_PATH
from core.llm import OllamaClient
from core.schemas import (
    ClaimStatus,
    ConflictResolutionDecision,
    ConflictType,
    ExtractionResponse,
    ExtractionResult,
    ExtractionValidationIssue,
    MemoryClaim,
    RawExtractedClaim,
)

EXTRACTION_SYSTEM_PROMPT = """You are RecallGuard Memory Extractor. Extract factual memory claims from conversations into structured JSON.

You must output a JSON object with a single key "claims" mapping to a list of claim objects.
Each claim object must have:
- "subject": the subject or entity (e.g., 'user', 'backend_system', 'project_deadline')
- "predicate": the property, preference, decision, or state (e.g., 'prefers_language', 'migration_target')
- "object_value": the specific value or target state (e.g., 'Go 1.22', 'dark mode')
- "source_quote": the EXACT verbatim sentence or phrase from the conversation text supporting this claim
- "confidence": number between 0.0 and 1.0

CRITICAL RULES:
1. "source_quote" MUST be copied word-for-word from the conversation text. Do NOT rephrase or fabricate quotes.
2. If no factual or persistent claims are found, return {"claims": []}.
3. Output ONLY valid JSON matching the format.

Example:
Input: "User: I prefer dark mode in all IDEs."
Output:
{
  "claims": [
    {
      "subject": "user",
      "predicate": "ide_theme_preference",
      "object_value": "dark mode",
      "source_quote": "I prefer dark mode in all IDEs.",
      "confidence": 0.95
    }
  ]
}"""


class MemoryService:
    """Service coordinating memory claims persistence, AI extraction, and conflict resolution."""

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        llm_client: Optional[OllamaClient] = None,
    ):
        self.db = db_manager or DatabaseManager(DEFAULT_DB_PATH)
        self.llm = llm_client or OllamaClient()

    def initialize(self) -> None:
        """Initialize database schema safely."""
        self.db.init_db()

    def save_claim(self, claim: MemoryClaim) -> MemoryClaim:
        """Store or update a memory claim in the database."""
        return self.db.insert_claim(claim)

    def save_claims(self, claims: List[MemoryClaim]) -> List[MemoryClaim]:
        """Store a batch of memory claims in the database."""
        saved = []
        for claim in claims:
            saved.append(self.save_claim(claim))
        return saved

    def get_claim(self, claim_id: str) -> Optional[MemoryClaim]:
        """Fetch a single claim by identifier."""
        return self.db.get_claim(claim_id)

    def list_claims(
        self, active_only: bool = False, status: Optional[str] = None
    ) -> List[MemoryClaim]:
        """Retrieve stored memory claims with optional status filtering."""
        return self.db.list_claims(active_only=active_only, status=status)

    def get_stats(self) -> dict:
        """Return memory store statistics including status breakdown."""
        return self.db.count_claims_detailed()


    # --- Feature 02: AI Memory Extraction & Verification ---

    @staticmethod
    def verify_quote_in_text(source_quote: str, text: str) -> bool:
        """Verify that the source quote actually appears verbatim in the input text."""
        if not source_quote or not source_quote.strip():
            return False
        norm_quote = " ".join(source_quote.strip().split()).lower()
        norm_text = " ".join(text.split()).lower()
        return norm_quote in norm_text

    @staticmethod
    def _extract_json_block(raw_text: str) -> str:
        """Extract JSON string from potential markdown code fences or raw text."""
        cleaned = raw_text.strip()
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
        if json_match:
            return json_match.group(1).strip()
        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            return cleaned[first_brace : last_brace + 1]
        return cleaned

    def extract_claims_from_text(
        self, text: str, session_id: Optional[str] = None
    ) -> ExtractionResult:
        """Extract atomic memory claims from conversation text using local Ollama.
        
        Validates model output structure via Pydantic and enforces that each
        claim's source quote exists verbatim in the source conversation.
        """
        if not text or not text.strip():
            return ExtractionResult(
                valid_claims=[],
                rejected_claims=[],
                raw_model_output="",
                success=True,
            )

        try:
            raw_output = self.llm.chat_structured(
                prompt=f"Extract memory claims from this conversation:\n\n{text}",
                system_prompt=EXTRACTION_SYSTEM_PROMPT,
                format_type="json",
                think=False,
            )
        except Exception as exc:
            return ExtractionResult(
                valid_claims=[],
                rejected_claims=[],
                raw_model_output="",
                success=False,
                error_message=f"Local Ollama inference failed: {str(exc)}",
            )

        if not raw_output or not raw_output.strip():
            return ExtractionResult(
                valid_claims=[],
                rejected_claims=[],
                raw_model_output=raw_output,
                success=False,
                error_message="Ollama returned an empty response.",
            )

        json_str = self._extract_json_block(raw_output)
        try:
            parsed_data = json.loads(json_str)
        except Exception as json_err:
            return ExtractionResult(
                valid_claims=[],
                rejected_claims=[
                    ExtractionValidationIssue(
                        claim_data={"raw_text": raw_output},
                        reason=f"Malformed model output could not be parsed as JSON: {str(json_err)}",
                    )
                ],
                raw_model_output=raw_output,
                success=False,
                error_message=f"JSON decoding error: {str(json_err)}",
            )

        raw_items: List[Any] = []
        if isinstance(parsed_data, dict):
            raw_items = parsed_data.get("claims", [])
        elif isinstance(parsed_data, list):
            raw_items = parsed_data
        else:
            return ExtractionResult(
                valid_claims=[],
                rejected_claims=[
                    ExtractionValidationIssue(
                        claim_data={"parsed": parsed_data},
                        reason="Expected JSON object with 'claims' list or top-level list.",
                    )
                ],
                raw_model_output=raw_output,
                success=False,
                error_message="Unexpected JSON structure: top-level must be object or list.",
            )

        valid_claims: List[MemoryClaim] = []
        rejected_claims: List[ExtractionValidationIssue] = []

        for item in raw_items:
            if not isinstance(item, dict):
                rejected_claims.append(
                    ExtractionValidationIssue(
                        claim_data={"item": str(item)},
                        reason="Claim entry must be a JSON object dictionary.",
                    )
                )
                continue

            try:
                candidate = RawExtractedClaim.model_validate(item)
            except ValidationError as val_err:
                rejected_claims.append(
                    ExtractionValidationIssue(
                        claim_data=item,
                        reason=f"Pydantic schema validation error: {val_err.errors()[0]['msg']}",
                    )
                )
                continue

            if not self.verify_quote_in_text(candidate.source_quote, text):
                rejected_claims.append(
                    ExtractionValidationIssue(
                        claim_data=item,
                        reason=f"Unsupported source quote: verbatim text '{candidate.source_quote}' was not found in the input conversation.",
                    )
                )
                continue

            claim = MemoryClaim(
                subject=candidate.subject,
                predicate=candidate.predicate,
                object_value=candidate.object_value,
                confidence=candidate.confidence,
                source_quote=candidate.source_quote,
                source_session_id=session_id,
                valid_from=candidate.valid_from,
                valid_until=candidate.valid_until,
                status=ClaimStatus.ACTIVE.value,
                is_active=True,
                metadata={
                    "source_quote": candidate.source_quote,
                    "extracted_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            valid_claims.append(claim)

        return ExtractionResult(
            valid_claims=valid_claims,
            rejected_claims=rejected_claims,
            raw_model_output=raw_output,
            success=True,
        )

    # --- Feature 03: Temporal Memory Conflict Resolution ---

    def resolve_conflicts(
        self,
        new_claim: MemoryClaim,
        conversation_context: Optional[str] = None,
    ) -> ConflictResolutionDecision:
        """Compare a proposed memory claim against existing active records.
        
        Uses deterministic business rules and quote evidence to classify:
        - Exact duplicate
        - Conflicting value with explicit update evidence
        - Ambiguous conflict without update evidence
        - Invalid evidence
        - No conflict
        """
        # Look for existing active claims matching the subject and predicate
        matching_actives = self.db.find_matching_claims(
            subject=new_claim.subject,
            predicate=new_claim.predicate,
            active_only=True,
        )

        return evaluate_claim_conflict(
            proposed_claim=new_claim,
            existing_claims=matching_actives,
            conversation_context=conversation_context,
        )

    def process_and_stage_claim(
        self,
        proposed_claim: MemoryClaim,
        auto_supersede_explicit: bool = False,
    ) -> Tuple[MemoryClaim, ConflictResolutionDecision]:
        """Evaluate conflict and stage or persist claim according to rules.
        
        - If NO_CONFLICT: saved as active.
        - If EXACT_DUPLICATE: returns existing active claim without duplicate entry.
        - If EXPLICIT_UPDATE:
            - If auto_supersede_explicit=True: atomically supersedes old claim.
            - Else: saved as 'needs_review' with supersedes_id link.
        - If AMBIGUOUS_CHANGE or INVALID_EVIDENCE:
            - Saved as 'needs_review'.
        """
        decision = self.resolve_conflicts(proposed_claim)

        if decision.conflict_type == ConflictType.EXACT_DUPLICATE.value and decision.existing_claim:
            return decision.existing_claim, decision

        if decision.conflict_type == ConflictType.NO_CONFLICT.value:
            proposed_claim.status = ClaimStatus.ACTIVE.value
            proposed_claim.is_active = True
            saved = self.save_claim(proposed_claim)
            return saved, decision

        if decision.conflict_type == ConflictType.EXPLICIT_UPDATE.value and decision.existing_claim:
            if auto_supersede_explicit:
                superseded_claim = self.approve_conflict_update(
                    proposed_claim=proposed_claim,
                    existing_claim_id=decision.existing_claim.id,
                )
                return superseded_claim, decision
            else:
                # Stage as needs_review for explicit user approval
                proposed_claim.status = ClaimStatus.NEEDS_REVIEW.value
                proposed_claim.is_active = False
                proposed_claim.supersedes_id = decision.existing_claim.id
                proposed_claim.metadata["conflict_reason"] = decision.decision_reason
                proposed_claim.metadata["conflict_type"] = decision.conflict_type
                saved = self.save_claim(proposed_claim)
                return saved, decision

        # For AMBIGUOUS_CHANGE or INVALID_EVIDENCE
        proposed_claim.status = ClaimStatus.NEEDS_REVIEW.value
        proposed_claim.is_active = False
        if decision.existing_claim:
            proposed_claim.supersedes_id = decision.existing_claim.id
        proposed_claim.metadata["conflict_reason"] = decision.decision_reason
        proposed_claim.metadata["conflict_type"] = decision.conflict_type
        saved = self.save_claim(proposed_claim)
        return saved, decision

    def approve_conflict_update(
        self, proposed_claim: MemoryClaim, existing_claim_id: str
    ) -> MemoryClaim:
        """Approve an update: atomically supersede the old record and activate the replacement."""
        return self.db.supersede_claim_atomic(
            old_claim_id=existing_claim_id,
            new_claim=proposed_claim,
        )

    def reject_conflict_update(
        self, proposed_claim_id: str, reason: str = "Rejected during conflict review"
    ) -> bool:
        """Reject an update: mark the proposed claim as superseded/rejected without touching existing active claim."""
        return self.db.update_claim_status(
            claim_id=proposed_claim_id,
            status=ClaimStatus.SUPERSEDED.value,
            is_active=False,
            reason=reason,
        )

    def list_review_queue(self) -> List[Dict[str, Any]]:
        """Return list of claims pending review with their corresponding existing active claims."""
        pending_claims = self.list_claims(status=ClaimStatus.NEEDS_REVIEW.value)
        queue = []
        for pending in pending_claims:
            existing: Optional[MemoryClaim] = None
            if pending.supersedes_id:
                existing = self.get_claim(pending.supersedes_id)
            if not existing:
                # Search by subject and predicate
                matches = self.db.find_matching_claims(
                    subject=pending.subject, predicate=pending.predicate, active_only=True
                )
                if matches:
                    existing = matches[0]

            decision = evaluate_claim_conflict(pending, [existing] if existing else [])
            queue.append(
                {
                    "pending_claim": pending,
                    "existing_claim": existing,
                    "decision": decision,
                }
            )
        return queue

    # --- Foundation Stubs for Upcoming Milestones ---

    def retrieve_reliable_memories(self, query: str) -> List[MemoryClaim]:
        """Stub: Temporal reliability retrieval will be implemented in Feature 04."""
        raise NotImplementedError(
            "Reliability retrieval logic is scheduled for implementation in Feature 04."
        )

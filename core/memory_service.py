"""Memory Service for managing claims, storage, and foundational lifecycle operations."""

from datetime import datetime, timezone
import json
import re
from typing import Any, Dict, List, Optional
from pydantic import ValidationError

from core.database import DatabaseManager, DEFAULT_DB_PATH
from core.llm import OllamaClient
from core.schemas import (
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
    """Service coordinating memory claims persistence, AI extraction, and verification.
    
    Milestone 02 implements local AI claim extraction, Pydantic validation,
    and verbatim source quote verification.
    """

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

    def list_claims(self, active_only: bool = False) -> List[MemoryClaim]:
        """Retrieve stored memory claims."""
        return self.db.list_claims(active_only=active_only)

    def get_stats(self) -> dict:
        """Return memory store statistics."""
        return self.db.count_claims()

    # --- Feature 02: AI Memory Extraction & Verification ---

    @staticmethod
    def verify_quote_in_text(source_quote: str, text: str) -> bool:
        """Verify that the source quote actually appears verbatim in the input text."""
        if not source_quote or not source_quote.strip():
            return False
        # Normalize whitespace and compare case-insensitively
        norm_quote = " ".join(source_quote.strip().split()).lower()
        norm_text = " ".join(text.split()).lower()
        return norm_quote in norm_text

    @staticmethod
    def _extract_json_block(raw_text: str) -> str:
        """Extract JSON string from potential markdown code fences or raw text."""
        cleaned = raw_text.strip()
        # Look for ```json ... ``` blocks
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
        if json_match:
            return json_match.group(1).strip()
        # Look for outermost { ... } or [ ... ]
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

        # 1. Call local Ollama inference
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

        # 2. Parse JSON
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

        # 3. Handle structure (either {"claims": [...]} or raw list [...])
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

        # 4. Validate each item with Pydantic and verify source quote
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

            # Verify that source_quote actually appears in the input conversation
            if not self.verify_quote_in_text(candidate.source_quote, text):
                rejected_claims.append(
                    ExtractionValidationIssue(
                        claim_data=item,
                        reason=f"Unsupported source quote: verbatim text '{candidate.source_quote}' was not found in the input conversation.",
                    )
                )
                continue

            # Create validated MemoryClaim
            claim = MemoryClaim(
                subject=candidate.subject,
                predicate=candidate.predicate,
                object_value=candidate.object_value,
                confidence=candidate.confidence,
                source_quote=candidate.source_quote,
                source_session_id=session_id,
                valid_from=candidate.valid_from,
                valid_until=candidate.valid_until,
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

    # --- Foundation Stubs for Upcoming Milestones ---

    def resolve_conflicts(self, new_claim: MemoryClaim) -> List[MemoryClaim]:
        """Stub: Temporal conflict resolution will be implemented in Feature 03."""
        raise NotImplementedError(
            "Conflict resolution logic is scheduled for implementation in Feature 03."
        )

    def retrieve_reliable_memories(self, query: str) -> List[MemoryClaim]:
        """Stub: Temporal reliability retrieval will be implemented in Feature 04."""
        raise NotImplementedError(
            "Reliability retrieval logic is scheduled for implementation in Feature 04."
        )

"""Evidence-Grounded Retrieval Service for RecallGuard (Feature 04).

Searches the SQLite memory store for claims relevant to a user question,
filters out superseded and expired records, and generates a grounded answer
using local Ollama that cites only the retrieved evidence.

Cloud APIs (OpenAI, Gemini, Anthropic, etc.) are strictly prohibited.
All inference is performed locally via Ollama with qwen3.5:0.8b.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from core.database import DatabaseManager
from core.llm import OllamaClient
from core.schemas import ClaimStatus, MemoryClaim


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class RetrievalResult(BaseModel):
    """Evidence bundle returned by the retrieval service for a single query."""

    query: str
    retrieved_claims: List[MemoryClaim] = Field(default_factory=list)
    filtered_out_count: int = 0          # superseded / expired / needs_review
    answer: str = ""
    abstained: bool = False
    abstention_reason: str = ""
    evidence_ids: List[str] = Field(default_factory=list)    # IDs cited in answer
    evidence_quotes: List[str] = Field(default_factory=list)  # quotes cited
    confidence: float = 0.0              # mean confidence of retrieved evidence


class RetrievalConfig(BaseModel):
    """Knobs for retrieval behaviour."""

    max_results: int = Field(default=10, ge=1, le=50)
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    include_needs_review: bool = False   # expose unresolved conflicts to answer
    keyword_match_threshold: int = Field(default=1, ge=1)  # minimum keyword hits


# ---------------------------------------------------------------------------
# Domain taxonomy for mismatch penalty
# ---------------------------------------------------------------------------

# Maps a domain label to the set of tokens that identify it.
_DOMAIN_TOKENS: Dict[str, Set[str]] = {
    "framework": {"framework", "frameworks", "stack", "tech_stack", "backend_framework",
                  "frontend", "runtime", "server", "web_framework"},
    "database":  {"database", "databases", "database_engine", "database_changes", "db",
                  "postgresql", "postgres", "mongodb", "mongo", "mysql", "sqlite",
                  "redis", "cassandra", "dynamodb", "storage"},
    "language":  {"language", "languages", "preferred_language", "python", "java",
                  "golang", "go", "javascript", "typescript", "ruby", "rust",
                  "kotlin", "swift", "cpp", "csharp"},
    "theme":     {"theme", "ide_theme", "dark", "light", "colour_scheme", "color_scheme"},
    "deadline":  {"deadline", "release", "schedule", "due_date", "milestone", "sprint"},
}

# Field scoring weights
_WEIGHT_PREDICATE = 5
_WEIGHT_SUBJECT   = 3
_WEIGHT_OBJECT    = 3
_WEIGHT_QUOTE     = 1
_DOMAIN_BOOST     = 10
_DOMAIN_MISMATCH  = -20

# Adaptive gap threshold: if top score >= this, apply gap filtering
_SCORE_GAP_ACTIVATION = 6
_SCORE_GAP_RATIO      = 0.5   # keep claims >= max_score * ratio (min 2)

# Confirmed completed migration / in-use markers in source_quote
_CONFIRMED_PATTERNS: Tuple[str, ...] = (
    "we migrated",
    "we have migrated",
    "team migrated",
    "team has migrated",
    "we switched to",
    "we have switched to",
    "has been migrated to",
    "have been migrated to",
    "was migrated to",
    "were migrated to",
    "migration is complete",
    "migration was completed",
    "migration is done",
    "successfully migrated",
    "completed migration",
    "completed the migration",
    "now running on",
    "now running",
    "now using",
    "currently running on",
    "currently running",
    "currently uses",
    "currently using",
    "is now using",
    "is now running",
)

# Planned / future migration markers detected in source_quote
_PLANNED_PATTERNS: Tuple[str, ...] = (
    "starting next month",
    "plans to migrate",
    "planning to migrate",
    "plan to migrate",
    "will migrate",
    "next month",
    "next quarter",
    "next year",
    "scheduled to",
    "scheduled for",
    "intends to migrate",
    "intend to migrate",
    "planning on migrating",
    "planning to switch",
    "plans to switch",
)

# Decision / intention markers in source_quote (migration decided, but completion NOT confirmed)
_DECISION_PATTERNS: Tuple[str, ...] = (
    "decided to migrate",
    "decided to move",
    "decided to switch",
    "decided on",
    "decided that",
    "decision to migrate",
    "chosen to migrate",
    "chose to migrate",
    "agreed to migrate",
    "agreed to move",
    "agreed to switch",
    "agreed on",
    "selected to migrate",
    "selected go",
    "resolved to migrate",
    "approved migration",
    "approved the migration",
    "we decided",
    "team decided",
    "opted to migrate",
    "intent to migrate",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "shall",
        "should", "may", "might", "must", "can", "could", "of", "in", "on",
        "at", "to", "for", "with", "by", "from", "or", "and", "not", "no",
        "what", "who", "when", "where", "which", "how", "tell", "me", "my",
        "about", "show", "list", "give", "get", "find", "about",
    }
)


def _tokenize(text: str) -> List[str]:
    """Return meaningful lowercase tokens (strips stop-words)."""
    tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


def _token_set(text: str) -> Set[str]:
    """Return a set of all lowercase alphanumeric tokens from *text*."""
    return set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))


def _detect_query_domain(query_tokens: List[str]) -> Optional[str]:
    """Identify the primary domain the query is asking about, if any."""
    query_set = set(query_tokens)
    for domain, domain_set in _DOMAIN_TOKENS.items():
        if query_set & domain_set:
            return domain
    return None


def _detect_claim_domain(claim: MemoryClaim) -> Optional[str]:
    """Identify the domain a claim belongs to based on its predicate and object_value."""
    claim_tokens = _token_set(
        " ".join(filter(None, [claim.predicate, claim.object_value]))
    )
    for domain, domain_set in _DOMAIN_TOKENS.items():
        if claim_tokens & domain_set:
            return domain
    return None


def _score_claim(claim: MemoryClaim, query_tokens: List[str]) -> int:
    """Return a weighted keyword hit-count for ranking retrieved claims.

    Scoring weights (per-field):
      predicate   : 5 pts per token match
      subject     : 3 pts per token match
      object_value: 3 pts per token match
      source_quote: 1 pt  per token match

    Domain behaviour:
      +10  if claim domain exactly matches query domain
      -20  if claim domain is distinct from (and incompatible with) query domain
    """
    query_set = set(query_tokens)

    pred_tokens   = _token_set(claim.predicate)
    subj_tokens   = _token_set(claim.subject)
    obj_tokens    = _token_set(claim.object_value)
    quote_tokens  = _token_set(claim.source_quote or "")

    score  = len(query_set & pred_tokens)  * _WEIGHT_PREDICATE
    score += len(query_set & subj_tokens)  * _WEIGHT_SUBJECT
    score += len(query_set & obj_tokens)   * _WEIGHT_OBJECT
    score += len(query_set & quote_tokens) * _WEIGHT_QUOTE

    # Domain alignment / mismatch adjustment
    query_domain = _detect_query_domain(query_tokens)
    claim_domain = _detect_claim_domain(claim)
    if query_domain is not None and claim_domain is not None:
        if query_domain == claim_domain:
            score += _DOMAIN_BOOST
        elif query_domain != claim_domain:
            score += _DOMAIN_MISMATCH

    return score


def _extract_root_entity(subject: str) -> str:
    """Normalise a subject string to its root entity token.

    e.g. 'backend_service' -> 'backend', 'backend_system' -> 'backend'
    """
    parts = re.split(r"[_\-\s]", subject.lower())
    return parts[0] if parts else subject.lower()


def _is_confirmed_completion(source_quote: Optional[str]) -> bool:
    """Return True if source_quote explicitly confirms migration/change is completed."""
    if not source_quote:
        return False
    q_lower = source_quote.lower()
    return any(pat in q_lower for pat in _CONFIRMED_PATTERNS)


def _is_planned_migration(source_quote: Optional[str]) -> bool:
    """Return True if *source_quote* contains planned/future migration language."""
    if not source_quote:
        return False
    q_lower = source_quote.lower()
    if _is_confirmed_completion(source_quote):
        return False
    return any(pat in q_lower for pat in _PLANNED_PATTERNS)


def _is_decision_only(source_quote: Optional[str]) -> bool:
    """Return True if source_quote indicates a migration decision but not completion."""
    if not source_quote:
        return False
    q_lower = source_quote.lower()
    if _is_confirmed_completion(source_quote):
        return False
    if _is_planned_migration(source_quote):
        return False
    return any(pat in q_lower for pat in _DECISION_PATTERNS)


def _classify_temporal(source_quote: Optional[str]) -> str:
    """Return 'confirmed', 'planned', 'decision_only', or 'factual'."""
    if not source_quote:
        return "factual"
    if _is_confirmed_completion(source_quote):
        return "confirmed"
    if _is_planned_migration(source_quote):
        return "planned"
    if _is_decision_only(source_quote):
        return "decision_only"
    return "factual"


def _question_asks_current_state(query: str) -> bool:
    """Return True if the question specifically asks what is currently in use or running."""
    q = query.lower()
    current_qualifiers = [
        "currently",
        "current",
        "presently",
        "right now",
        "today",
        "at the moment",
        "now",
    ]
    return any(re.search(r"\b" + re.escape(kw) + r"\b", q) for kw in current_qualifiers)


def _clean_grounded_answer(
    raw_answer: str,
    retrieved_claims: List[MemoryClaim],
) -> Tuple[str, bool, str]:
    """Clean model response, stripping spurious or contradictory ABSTAIN blocks.

    Returns:
        (answer_text, is_abstained, abstention_reason)
    """
    stripped = raw_answer.strip()
    if not stripped:
        return "", True, "Local model returned an empty response."

    # Case 1: Trailing ABSTAIN section after an actual answer.
    # e.g., "<answer text>\n\nABSTAIN: <caveat>"
    # If the text before ABSTAIN has substantive length or citations, strip the trailing ABSTAIN.
    parts = re.split(r"\n+(?=(?:ABSTAIN|Abstain):)", stripped)
    if len(parts) > 1 and len(parts[0].strip()) >= 20:
        cleaned = parts[0].strip()
        return cleaned, False, ""

    # Case 2: Model started with ABSTAIN, but then provided an actual substantive answer.
    # e.g., "ABSTAIN: The evidence indicates that the backend was migrated to Go 1.22..."
    # or "ABSTAIN: <reason>\n\nOur backend is Go 1.22 [E1]."
    if stripped.upper().startswith("ABSTAIN"):
        subsequent_text = re.sub(
            r"^(?:ABSTAIN|Abstain):\s*(?:<[^>]+>|[^\n]*\n+)?", "", stripped
        ).strip()
        has_claim_content = any(
            c.object_value.lower() in stripped.lower()
            for c in retrieved_claims
            if c.object_value
        )
        has_citation = bool(re.search(r"\[E\d+\]", stripped))
        has_migration_answer = any(
            kw in stripped.lower()
            for kw in ["migrated", "superseded", "currently running", "replaced", "current version"]
        )
        if (has_claim_content or has_citation or has_migration_answer) and len(subsequent_text) >= 20:
            return subsequent_text, False, ""
        return "", True, stripped

    # Case 3: Clean answer
    return stripped, False, ""


def _make_planned_migration_answer(claims: List[MemoryClaim]) -> str:
    """Build a deterministic, accurate answer when all evidence confirms only a planned migration.

    Called when the LLM model hallucinates a completed migration from future-tense source quotes.
    """
    parts: List[str] = []
    for i, c in enumerate(claims, start=1):
        quote = c.source_quote or ""
        parts.append(
            f"A migration to {c.object_value} is planned [E{i}] "
            f"(\"{quote}\"); current in-use status is not yet confirmed."
        )
    return " ".join(parts)


def _make_decision_answer(claims: List[MemoryClaim]) -> str:
    """Build a deterministic, accurate statement when evidence records only a migration decision."""
    parts: List[str] = []
    for i, c in enumerate(claims, start=1):
        quote = c.source_quote or ""
        parts.append(
            f"A migration to {c.object_value} was decided [E{i}] "
            f"(\"{quote}\"); whether the migration has been completed and {c.object_value} is currently in use is unconfirmed by the available evidence."
        )
    return " ".join(parts)


def _answer_implies_current_use(answer: str, value: str) -> bool:
    """Return True when *answer* asserts that *value* is currently in use."""
    val_lower = value.lower()
    ans_lower = answer.lower()
    if val_lower not in ans_lower:
        return False
    # Phrases that assert current use
    current_use_patterns = [
        r"\bcurrently\s+(?:uses?|running|using|in use|deployed|active)\b",
        r"\bactively\s+(?:uses?|using|running)\b",
        r"\bis\s+(?:now\s+)?(?:using|running|deployed)\b",
        r"\bhas\s+(?:already\s+)?migrated\b",
        r"\bmigration\s+(?:is\s+)?(?:complete|done|finished|successful)\b",
        r"\bnow\s+(?:running|using|deployed|on)\b",
    ]
    for pat in current_use_patterns:
        if re.search(pat, ans_lower):
            return True
    return False


# ---------------------------------------------------------------------------
# Main service
# ---------------------------------------------------------------------------

class RetrievalService:
    """Retrieves evidence-grounded memory claims and generates grounded answers.

    Workflow
    --------
    1. Tokenise the question and score all *active* claims by keyword overlap.
    2. Filter out expired, superseded, and needs_review (unless configured) claims.
    3. Rank by keyword hits DESC → confidence DESC → recency DESC.
    4. Apply adaptive score-gap filtering to drop low-relevance noise claims.
    5. Deduplicate overlapping evidence (same root entity + identical object_value).
    6. If too few claims pass the threshold → abstain with a clear reason.
    7. Build a grounded prompt for local Ollama and generate a cited answer.
    8. Post-process answer for temporal accuracy (planned vs. completed migration).
    """

    GROUNDED_SYSTEM_PROMPT = (
        "You are RecallGuard, a memory reliability assistant. "
        "Answer the user's question directly and concisely using ONLY the provided memory evidence below. "
        "Do NOT invent facts that are absent from the evidence. "
        "IMPORTANT TEMPORAL RULES — you MUST follow these exactly:\n"
        "  1. If a source_quote says a migration is 'starting next month', 'plans to migrate', "
        "'will migrate', 'next month', 'next quarter', or any other future-tense phrasing, "
        "you MUST state that the migration is PLANNED and that the current in-use status is NOT YET CONFIRMED. "
        "Do NOT say the technology is currently used unless a separate source_quote confirms completed use.\n"
        "  2. If a source_quote says a migration was decided (e.g., 'we decided to migrate', 'decided to switch', "
        "'agreed to move') but does not confirm completion, you MUST explain that a migration decision is recorded "
        "but completion is unconfirmed. When the question specifically asks what is currently in use, state that "
        "current status is unconfirmed and ABSTAIN.\n"
        "  3. Only if a source_quote explicitly confirms completed migration (e.g., 'we migrated', 'we switched to', "
        "'now running', 'is running') may you state that the new technology is currently in use.\n"
        "  4. Never infer that a decision or planned migration has already completed.\n"
        "For every factual statement cite the Evidence ID (e.g., [E1]) and include the verbatim "
        "source_quote as supporting proof. "
        "Do NOT include an ABSTAIN section when the evidence provides a clear answer."
    )

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        llm_client: Optional[OllamaClient] = None,
        config: Optional[RetrievalConfig] = None,
    ) -> None:
        self.db = db_manager or DatabaseManager()
        self.llm = llm_client or OllamaClient()
        self.config = config or RetrievalConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> RetrievalResult:
        """Search memory store for claims relevant to *query*.

        Returns a :class:`RetrievalResult` with relevant claims filtered,
        ranked, deduplicated, and annotated, but without the LLM-generated answer.
        Use :meth:`answer` for the full end-to-end pipeline.
        """
        if not query or not query.strip():
            return RetrievalResult(
                query=query,
                abstained=True,
                abstention_reason="Empty query provided.",
            )

        query_tokens = _tokenize(query)
        if not query_tokens:
            return RetrievalResult(
                query=query,
                abstained=True,
                abstention_reason="Query contained only stop-words or punctuation.",
            )

        # 1. Fetch candidate claims (active, plus needs_review if configured)
        if self.config.include_needs_review:
            candidates = self.db.list_claims(active_only=True) + self.db.list_claims(
                status=ClaimStatus.NEEDS_REVIEW.value
            )
        else:
            candidates = self.db.list_claims(active_only=True)
        now = datetime.now(timezone.utc)

        eligible: List[Tuple[int, MemoryClaim]] = []
        filtered_out = 0

        for claim in candidates:
            # Guard: skip needs_review unless configured
            if claim.status == ClaimStatus.NEEDS_REVIEW.value and not self.config.include_needs_review:
                filtered_out += 1
                continue
            # Guard: skip expired
            if claim.is_expired(as_of=now):
                filtered_out += 1
                continue
            # Guard: minimum confidence
            if claim.confidence < self.config.min_confidence:
                filtered_out += 1
                continue

            score = _score_claim(claim, query_tokens)
            if score >= self.config.keyword_match_threshold:
                eligible.append((score, claim))

        # 2. Sort: keyword hits DESC → confidence DESC → created_at DESC
        eligible.sort(key=lambda t: (t[0], t[1].confidence, t[1].created_at), reverse=True)

        # 3. Adaptive score-gap filtering: drop noise when top score is clearly dominant
        if eligible:
            max_score = eligible[0][0]
            if max_score >= _SCORE_GAP_ACTIVATION:
                min_acceptable = max(2, int(max_score * _SCORE_GAP_RATIO))
                eligible = [(s, c) for s, c in eligible if s >= min_acceptable]

        # 4. Overlapping-evidence deduplication:
        #    Skip a claim if a higher-ranked, already-selected claim has the same
        #    (root_entity, object_value) pair OR the same (subject, predicate) pair.
        seen_entity_value: Set[Tuple[str, str]] = set()
        seen_subject_predicate: Set[Tuple[str, str]] = set()
        top_claims: List[MemoryClaim] = []

        for score, claim in eligible:
            if len(top_claims) >= self.config.max_results:
                break

            root = _extract_root_entity(claim.subject)
            val  = claim.object_value.strip().lower()
            ev_key = (root, val)
            sp_key = (claim.subject.lower(), claim.predicate.lower())

            if ev_key in seen_entity_value or sp_key in seen_subject_predicate:
                # Duplicate / overlapping evidence — skip (claim remains in DB)
                continue

            seen_entity_value.add(ev_key)
            seen_subject_predicate.add(sp_key)
            top_claims.append(claim)

        mean_conf = (
            sum(c.confidence for c in top_claims) / len(top_claims) if top_claims else 0.0
        )

        return RetrievalResult(
            query=query,
            retrieved_claims=top_claims,
            filtered_out_count=filtered_out,
            abstained=len(top_claims) == 0,
            abstention_reason=(
                "No active memory claims matched the query." if not top_claims else ""
            ),
            evidence_ids=[c.id for c in top_claims],
            evidence_quotes=[c.source_quote for c in top_claims if c.source_quote],
            confidence=round(mean_conf, 4),
        )

    def answer(self, query: str) -> RetrievalResult:
        """End-to-end pipeline: retrieve → filter → Ollama grounded answer."""
        result = self.retrieve(query)

        # Abstain immediately when retrieval found nothing
        if result.abstained:
            return result

        # Check temporal classifications of retrieved claims
        temporal_types = [_classify_temporal(c.source_quote) for c in result.retrieved_claims]
        all_decision = bool(result.retrieved_claims) and all(t == "decision_only" for t in temporal_types)
        all_planned = bool(result.retrieved_claims) and all(t == "planned" for t in temporal_types)
        asks_current = _question_asks_current_state(query)

        # -----------------------------------------------------------------------
        # Temporal Decision Guard:
        # If the question asks what is currently in use or running, but evidence
        # only records a migration decision, RecallGuard MUST NOT assert current
        # use; it should explain that a migration decision is recorded but completion
        # is unconfirmed, and safely abstain.
        # -----------------------------------------------------------------------
        if all_decision and asks_current:
            sample_claim = result.retrieved_claims[0]
            val = sample_claim.object_value
            quote = sample_claim.source_quote or ""
            abstention_msg = (
                f"A migration decision to {val} is recorded [E1] (\"{quote}\"), "
                f"but completion is unconfirmed. "
                f"Cannot assert that {val} is currently in use."
            )
            result.abstained = True
            result.abstention_reason = abstention_msg
            result.answer = ""
            result.evidence_ids = [c.id for c in result.retrieved_claims]
            result.evidence_quotes = [c.source_quote for c in result.retrieved_claims if c.source_quote]
            return result

        # Build evidence block with temporal status annotations
        evidence_block = self._build_evidence_block(result.retrieved_claims)

        temporal_instruction = ""
        if all_planned and result.retrieved_claims:
            sample_val = result.retrieved_claims[0].object_value
            temporal_instruction = (
                f"\nCRITICAL: Every source quote in the evidence below uses future/planning language. "
                f"You MUST NOT say that '{sample_val}' is currently used. "
                f"State only that the migration is planned."
            )
        elif all_decision and result.retrieved_claims:
            sample_val = result.retrieved_claims[0].object_value
            temporal_instruction = (
                f"\nCRITICAL: Every source quote in the evidence below records only a decision, not completed use. "
                f"You MUST NOT say that '{sample_val}' is currently used. "
                f"State that the migration was decided but completion is unconfirmed."
            )

        prompt = (
            f"User question: {query}\n\n"
            f"Memory Evidence:\n{evidence_block}\n\n"
            f"Provide a clear, grounded answer using the memory evidence above. "
            "Cite evidence IDs (e.g., [E1]) with source quotes. "
            "Obey the temporal rules in your system instructions precisely: "
            "if a quote says 'starting next month' or similar future language, state the migration is planned "
            "and current usage is not yet confirmed. "
            "If a quote says a migration decision was recorded (e.g. 'we decided to migrate') but not completed, "
            "explain the decision is recorded but completion is unconfirmed. "
            "Only state the technology is currently in use if a quote explicitly confirms completed migration. "
            f"Do not include an ABSTAIN section when evidence is provided.{temporal_instruction}"
        )

        try:
            raw_answer = self.llm.generate(
                prompt=prompt,
                system=self.GROUNDED_SYSTEM_PROMPT,
                options={"temperature": 0.1, "num_predict": 512},
            )
        except Exception as exc:
            result.abstained = True
            result.abstention_reason = f"Ollama inference failed: {exc}"
            result.evidence_ids = [c.id for c in result.retrieved_claims]
            result.evidence_quotes = [c.source_quote for c in result.retrieved_claims if c.source_quote]
            return result

        if not raw_answer or not raw_answer.strip():
            result.abstained = True
            result.abstention_reason = "Local model returned an empty response."
            result.evidence_ids = [c.id for c in result.retrieved_claims]
            result.evidence_quotes = [c.source_quote for c in result.retrieved_claims if c.source_quote]
            return result

        cleaned_answer, is_abstained, abstention_reason = _clean_grounded_answer(
            raw_answer, result.retrieved_claims
        )

        if is_abstained:
            result.abstained = True
            result.abstention_reason = abstention_reason
            result.answer = ""
            result.evidence_ids = [c.id for c in result.retrieved_claims]
            result.evidence_quotes = [c.source_quote for c in result.retrieved_claims if c.source_quote]
            return result

        # ---------------------------------------------------------------------------
        # Temporal post-processing: catch LLM hallucinations that claim
        # a planned or decision-only migration is completed / currently in use.
        # ---------------------------------------------------------------------------
        if all_planned and result.retrieved_claims:
            for claim in result.retrieved_claims:
                if _answer_implies_current_use(cleaned_answer, claim.object_value):
                    cleaned_answer = _make_planned_migration_answer(result.retrieved_claims)
                    break
        elif all_decision and result.retrieved_claims:
            for claim in result.retrieved_claims:
                if _answer_implies_current_use(cleaned_answer, claim.object_value):
                    cleaned_answer = _make_decision_answer(result.retrieved_claims)
                    break

        # Extract cited IDs from "[E<n>]" markers
        cited_indices = [int(m) - 1 for m in re.findall(r"\[E(\d+)\]", cleaned_answer)]
        cited_claims = [
            result.retrieved_claims[i]
            for i in cited_indices
            if 0 <= i < len(result.retrieved_claims)
        ]

        result.answer = cleaned_answer
        selected_claims = cited_claims if cited_claims else result.retrieved_claims
        result.evidence_ids = [c.id for c in selected_claims]
        result.evidence_quotes = [c.source_quote for c in selected_claims if c.source_quote]
        return result

    def get_active_claim_by_subject_predicate(
        self, subject: str, predicate: str
    ) -> Optional[MemoryClaim]:
        """Return the most recent active claim for a given subject/predicate pair."""
        matches = self.db.find_matching_claims(subject=subject, predicate=predicate, active_only=True)
        now = datetime.now(timezone.utc)
        for claim in matches:
            if not claim.is_expired(as_of=now):
                return claim
        return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_evidence_block(claims: List[MemoryClaim]) -> str:
        """Render an ordered evidence block with IDs and temporal status for the LLM prompt."""
        if not claims:
            return ""
        lines: List[str] = []
        for i, claim in enumerate(claims, start=1):
            quote = claim.source_quote or "N/A"
            temp_type = _classify_temporal(claim.source_quote)
            if temp_type == "planned":
                temporal_status = (
                    "PLANNED_MIGRATION — future tense; current in-use status NOT YET CONFIRMED"
                )
            elif temp_type == "decision_only":
                temporal_status = (
                    "DECISION_RECORDED — migration decided but completion NOT confirmed; current in-use status UNKNOWN"
                )
            else:
                temporal_status = "confirmed / completed"
            lines.append(
                f"[E{i}] id={claim.id[:8]}…\n"
                f"  subject: {claim.subject}\n"
                f"  predicate: {claim.predicate}\n"
                f"  value: {claim.object_value}\n"
                f"  confidence: {claim.confidence:.2f}\n"
                f"  source_quote: \"{quote}\"\n"
                f"  temporal_status: {temporal_status}\n"
                f"  status: {claim.status}\n"
                f"  created: {claim.created_at}"
            )
        return "\n".join(lines)

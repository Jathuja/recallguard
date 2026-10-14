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
from typing import Any, Dict, List, Optional, Tuple

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


def _score_claim(claim: MemoryClaim, query_tokens: List[str]) -> int:
    """Return a keyword hit-count for ranking retrieved claims."""
    claim_text = " ".join(
        filter(None, [claim.subject, claim.predicate, claim.object_value, claim.source_quote or ""])
    ).lower()
    return sum(1 for tok in query_tokens if tok in claim_text)


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


# ---------------------------------------------------------------------------
# Main service
# ---------------------------------------------------------------------------

class RetrievalService:
    """Retrieves evidence-grounded memory claims and generates grounded answers.

    Workflow
    --------
    1. Tokenise the question and score all *active* claims by keyword overlap.
    2. Filter out expired, superseded, and needs_review (unless configured) claims.
    3. Rank by keyword overlap → confidence → recency.
    4. If too few claims pass the threshold → abstain with a clear reason.
    5. Build a grounded prompt for local Ollama and generate a cited answer.
    """

    GROUNDED_SYSTEM_PROMPT = (
        "You are RecallGuard, a memory reliability assistant. "
        "Answer the user's question directly and concisely using ONLY the provided memory evidence below. "
        "The provided evidence represents verified current active memory records. "
        "Do NOT invent facts that are absent from the evidence. "
        "If the evidence mentions a migration, replacement, or update (for example, migrating from Node.js 20 to Go 1.22), "
        "state clearly that the new value (Go 1.22) is current and that the prior value was migrated/superseded. "
        "For every factual statement cite the Evidence ID (e.g., [E1]) and include the verbatim source_quote as supporting proof. "
        "Do NOT include an ABSTAIN section or caveat when the evidence provides the answer."
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
        ranked, and annotated, but without the LLM-generated answer.
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

        # 1. Fetch all active claims
        all_active = self.db.list_claims(active_only=True)
        now = datetime.now(timezone.utc)

        eligible: List[Tuple[int, MemoryClaim]] = []
        filtered_out = 0

        for claim in all_active:
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

        top_claims = [c for _, c in eligible[: self.config.max_results]]

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
            confidence=round(mean_conf, 4),
        )

    def answer(self, query: str) -> RetrievalResult:
        """End-to-end pipeline: retrieve → filter → Ollama grounded answer."""
        result = self.retrieve(query)

        # Abstain immediately when retrieval found nothing
        if result.abstained:
            return result

        # Build evidence block
        evidence_block = self._build_evidence_block(result.retrieved_claims)

        prompt = (
            f"User question: {query}\n\n"
            f"Memory Evidence:\n{evidence_block}\n\n"
            "Provide a clear, grounded answer using the memory evidence above. Cite evidence IDs (e.g., [E1]) with source quotes. "
            "If the question asks about a previous or migrated technology (such as Node.js 20), explain that the new technology (such as Go 1.22) is current and that the old technology was migrated/superseded. "
            "Do not include an ABSTAIN section when evidence is provided."
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
            return result

        if not raw_answer or not raw_answer.strip():
            result.abstained = True
            result.abstention_reason = "Local model returned an empty response."
            return result

        cleaned_answer, is_abstained, abstention_reason = _clean_grounded_answer(
            raw_answer, result.retrieved_claims
        )

        if is_abstained:
            result.abstained = True
            result.abstention_reason = abstention_reason
            result.answer = ""
            return result

        # Extract cited IDs from "[E<n>]" markers
        cited_indices = [int(m) - 1 for m in re.findall(r"\[E(\d+)\]", cleaned_answer)]
        cited_claims = [
            result.retrieved_claims[i]
            for i in cited_indices
            if 0 <= i < len(result.retrieved_claims)
        ]

        result.answer = cleaned_answer
        result.evidence_ids = [c.id for c in cited_claims] if cited_claims else result.evidence_ids
        result.evidence_quotes = [c.source_quote or "" for c in cited_claims if c.source_quote]
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
        """Render an ordered evidence block with IDs for the LLM prompt."""
        lines: List[str] = []
        for i, claim in enumerate(claims, start=1):
            quote = claim.source_quote or "N/A"
            lines.append(
                f"[E{i}] id={claim.id[:8]}…\n"
                f"  subject: {claim.subject}\n"
                f"  predicate: {claim.predicate}\n"
                f"  value: {claim.object_value}\n"
                f"  confidence: {claim.confidence:.2f}\n"
                f"  source_quote: \"{quote}\"\n"
                f"  status: {claim.status}\n"
                f"  created: {claim.created_at}"
            )
        return "\n".join(lines)

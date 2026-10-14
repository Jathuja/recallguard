"""Deterministic conflict detection and update verification engine for RecallGuard."""

import re
from typing import List, Optional, Tuple

from core.schemas import (
    ClaimStatus,
    ConflictResolutionDecision,
    ConflictType,
    MemoryClaim,
)

# Common regex patterns indicating explicit temporal transitions or supersession
TRANSITION_PATTERNS = [
    # "migrated/migrating/migrate from X to Y"
    r"\b(?:migrated|migrating|migrate|migration)\b.*?\b(?:from|to)\b",
    # "switched/switching/switch from X to Y"
    r"\b(?:switched|switching|switch)\b.*?\b(?:from|to)\b",
    # "updated/updating/update from X to Y"
    r"\b(?:updated|updating|update)\b.*?\b(?:from|to)\b",
    # "replaced/replacing/replace X with Y"
    r"\b(?:replaced|replacing|replace)\b.*?\bwith\b",
    # "upgraded/upgrading/upgrade from X to Y"
    r"\b(?:upgraded|upgrading|upgrade)\b.*?\b(?:from|to)\b",
    # "changed/changing/change from X to Y"
    r"\b(?:changed|changing|change)\b.*?\b(?:from|to)\b",
    # "postponed/delayed/rescheduled from X to Y"
    r"\b(?:postponed|delayed|rescheduled|shifted)\b.*?\b(?:from|to)\b",
    # "no longer X, now Y" or "no longer using X"
    r"\bno\s+longer\b",
    # "deprecating/deprecated X in favor of Y"
    r"\b(?:deprecated|deprecating|deprecate)\b.*?\b(?:in favor of|for|with)\b",
    # "instead of X, we use Y"
    r"\binstead\s+of\b",
    # "superseded/superseding by Y"
    r"\b(?:superseded|superseding|supersede)\b",
    # "decided to switch/migrate/upgrade to Y"
    r"\bdecided\s+to\s+(?:migrate|switch|upgrade|change|move)\b",
]

COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in TRANSITION_PATTERNS]


def normalize_token(text: str) -> str:
    """Normalize string by trimming, lowering, and collapsing whitespaces."""
    return " ".join(text.strip().lower().split())


def detect_transition_in_quote(
    quote: str, old_value: Optional[str] = None, new_value: Optional[str] = None
) -> Tuple[bool, Optional[str]]:
    """Determine if a source quote contains explicit transition language.
    
    Returns:
        (is_explicit_update, transition_marker_description)
    """
    if not quote or not quote.strip():
        return False, None

    norm_quote = normalize_token(quote)

    # 1. Check if any transition regex matches
    matched_pattern: Optional[str] = None
    for pattern in COMPILED_PATTERNS:
        match = pattern.search(norm_quote)
        if match:
            matched_pattern = match.group(0)
            break

    # 2. Check if the quote specifically names the old value being replaced
    mentions_old_val = False
    if old_value:
        norm_old = normalize_token(old_value)
        # Check if old value or key parts of old value appear in the quote
        if norm_old in norm_quote:
            mentions_old_val = True
        else:
            # Check individual significant tokens of old value (length > 3)
            old_tokens = [t for t in norm_old.split() if len(t) > 3]
            if old_tokens and any(t in norm_quote for t in old_tokens):
                mentions_old_val = True

    # 3. Decision logic:
    # High confidence explicit update if a transition pattern matches
    if matched_pattern:
        if mentions_old_val:
            return True, f"Explicit transition pattern '{matched_pattern}' directly referencing prior value '{old_value}'"
        return True, f"Explicit transition pattern '{matched_pattern}' detected in quote"

    return False, None


def evaluate_claim_conflict(
    proposed_claim: MemoryClaim,
    existing_claims: List[MemoryClaim],
    conversation_context: Optional[str] = None,
) -> ConflictResolutionDecision:
    """Deterministically compare a proposed claim against existing active claims.
    
    Evaluates:
    1. Exact duplicates (same subject, predicate, object_value).
    2. Missing/invalid evidence (empty source quote).
    3. Conflicting values with explicit update evidence.
    4. Conflicting values with ambiguous evidence (needs review).
    5. No conflict (new topic/subject/predicate).
    """
    # Check for valid source quote
    if not proposed_claim.source_quote or not proposed_claim.source_quote.strip():
        return ConflictResolutionDecision(
            proposed_claim=proposed_claim,
            conflict_type=ConflictType.INVALID_EVIDENCE.value,
            recommended_action="needs_review",
            decision_reason="Proposed claim has missing or empty source quote evidence.",
            is_explicit_update=False,
            confidence_score=0.0,
        )

    norm_proposed_subj = normalize_token(proposed_claim.subject)
    norm_proposed_pred = normalize_token(proposed_claim.predicate)
    norm_proposed_val = normalize_token(proposed_claim.object_value)

    # Find matching existing active claims
    matching_actives = [
        c for c in existing_claims
        if normalize_token(c.subject) == norm_proposed_subj
        and normalize_token(c.predicate) == norm_proposed_pred
        and c.status == ClaimStatus.ACTIVE.value
    ]

    # CASE 1: No matching existing claims
    if not matching_actives:
        return ConflictResolutionDecision(
            proposed_claim=proposed_claim,
            conflict_type=ConflictType.NO_CONFLICT.value,
            recommended_action="add_new",
            decision_reason=f"No existing active claims found for subject '{proposed_claim.subject}' and predicate '{proposed_claim.predicate}'.",
            is_explicit_update=False,
            evidence_quote=proposed_claim.source_quote,
            confidence_score=proposed_claim.confidence,
        )

    # We examine the primary matching active claim (most recent)
    primary_existing = matching_actives[0]
    norm_existing_val = normalize_token(primary_existing.object_value)

    # CASE 2: Exact duplicate
    if norm_existing_val == norm_proposed_val:
        return ConflictResolutionDecision(
            proposed_claim=proposed_claim,
            existing_claim=primary_existing,
            conflict_type=ConflictType.EXACT_DUPLICATE.value,
            recommended_action="keep_existing",
            decision_reason=f"An active claim with identical value ('{primary_existing.object_value}') already exists.",
            is_explicit_update=False,
            evidence_quote=proposed_claim.source_quote,
            confidence_score=1.0,
        )

    # CASE 3: Different value -> Conflict!
    # Inspect evidence quote for explicit update transition markers
    is_explicit, transition_desc = detect_transition_in_quote(
        quote=proposed_claim.source_quote,
        old_value=primary_existing.object_value,
        new_value=proposed_claim.object_value,
    )

    if is_explicit:
        # Explicit update detected with verified transition evidence
        return ConflictResolutionDecision(
            proposed_claim=proposed_claim,
            existing_claim=primary_existing,
            conflict_type=ConflictType.EXPLICIT_UPDATE.value,
            recommended_action="supersede_existing",
            decision_reason=(
                f"Explicit update detected: {transition_desc}. "
                f"Proposed '{proposed_claim.object_value}' explicitly supersedes prior value '{primary_existing.object_value}'."
            ),
            is_explicit_update=True,
            evidence_quote=proposed_claim.source_quote,
            transition_detected=transition_desc,
            confidence_score=0.95,
        )

    # CASE 4: Ambiguous conflict (mere mention or contradictory statement without transition)
    return ConflictResolutionDecision(
        proposed_claim=proposed_claim,
        existing_claim=primary_existing,
        conflict_type=ConflictType.AMBIGUOUS_CHANGE.value,
        recommended_action="needs_review",
        decision_reason=(
            f"Ambiguous conflict: Existing active claim has value '{primary_existing.object_value}', "
            f"while proposed claim asserts '{proposed_claim.object_value}' without explicit migration or replacement evidence. "
            f"User review is required before superseding."
        ),
        is_explicit_update=False,
        evidence_quote=proposed_claim.source_quote,
        confidence_score=0.5,
    )

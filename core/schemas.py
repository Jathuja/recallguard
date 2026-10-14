"""Core data models and schemas for RecallGuard."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field, field_validator, model_validator


class ClaimStatus(str, Enum):
    """Logical lifecycle states of a memory claim."""
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    NEEDS_REVIEW = "needs_review"


class ConflictType(str, Enum):
    """Categorization of memory conflict scenarios."""
    NO_CONFLICT = "no_conflict"
    EXACT_DUPLICATE = "exact_duplicate"
    EXPLICIT_UPDATE = "explicit_update"
    AMBIGUOUS_CHANGE = "ambiguous_change"
    INVALID_EVIDENCE = "invalid_evidence"


class MemoryClaim(BaseModel):
    """Represents an atomic memory claim with temporal validity and reliability attributes.
    
    A claim expresses a subject-predicate-object assertion extracted from agent
    interactions, tagged with temporal bounds, source verification quotes, logical lifecycle
    status, and confidence ratings.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the claim (UUID).",
    )
    subject: str = Field(
        ...,
        min_length=1,
        description="Entity or topic the claim is about (e.g., 'user', 'project_deadline').",
    )
    predicate: str = Field(
        ...,
        min_length=1,
        description="Relationship or property asserted (e.g., 'prefers_language', 'status').",
    )
    object_value: str = Field(
        ...,
        min_length=1,
        description="Value or target of the assertion (e.g., 'Python', 'deprecated').",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for this claim between 0.0 and 1.0.",
    )
    source_quote: Optional[str] = Field(
        default=None,
        description="Verbatim quote from conversation text directly supporting this claim.",
    )
    source_session_id: Optional[str] = Field(
        default=None,
        description="Identifier of the conversation or session where this claim originated.",
    )
    valid_from: Optional[datetime] = Field(
        default=None,
        description="Timestamp indicating when this claim became valid.",
    )
    valid_until: Optional[datetime] = Field(
        default=None,
        description="Timestamp indicating when this claim expires or becomes superseded.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the claim record was created in the system.",
    )
    status: str = Field(
        default=ClaimStatus.ACTIVE.value,
        description="Logical lifecycle status: 'active', 'superseded', or 'needs_review'.",
    )
    is_active: bool = Field(
        default=True,
        description="Whether this claim is currently active and eligible for retrieval.",
    )
    superseded_by: Optional[str] = Field(
        default=None,
        description="UUID of newer claim that superseded this claim.",
    )
    supersedes_id: Optional[str] = Field(
        default=None,
        description="UUID of older claim that this claim superseded.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary additional context, raw snippet, or tag information.",
    )

    @field_validator("subject", "predicate", "object_value", mode="before")
    @classmethod
    def strip_whitespace(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.strip()
        return v

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: Any) -> str:
        valid_statuses = {s.value for s in ClaimStatus}
        if isinstance(v, str) and v.lower() in valid_statuses:
            return v.lower()
        raise ValueError(f"Status must be one of {valid_statuses}")

    @model_validator(mode="before")
    @classmethod
    def sync_status_and_is_active(cls, data: Any) -> Any:
        """Synchronize status and is_active for backward compatibility."""
        if isinstance(data, dict):
            # If is_active is explicitly False and status not specified, mark superseded
            if data.get("is_active") is False and "status" not in data:
                data["status"] = ClaimStatus.SUPERSEDED.value
            elif data.get("status") in (ClaimStatus.SUPERSEDED.value, ClaimStatus.NEEDS_REVIEW.value) and "is_active" not in data:
                data["is_active"] = False
            elif data.get("status") == ClaimStatus.ACTIVE.value and "is_active" not in data:
                data["is_active"] = True
            elif data.get("is_active") is False and data.get("status") == ClaimStatus.ACTIVE.value:
                data["status"] = ClaimStatus.SUPERSEDED.value
        return data

    def is_expired(self, as_of: Optional[datetime] = None) -> bool:
        """Check if this claim is expired as of a given timestamp."""
        check_time = as_of or datetime.now(timezone.utc)
        if self.valid_until is not None:
            if self.valid_until.tzinfo is not None and check_time.tzinfo is None:
                check_time = check_time.replace(tzinfo=timezone.utc)
            elif self.valid_until.tzinfo is None and check_time.tzinfo is not None:
                check_time = check_time.astimezone(timezone.utc).replace(tzinfo=None)
            return check_time > self.valid_until
        return False


class ConflictResolutionDecision(BaseModel):
    """Outcome and explanation of comparing a proposed claim against existing memories."""

    proposed_claim: MemoryClaim
    existing_claim: Optional[MemoryClaim] = None
    conflict_type: str = ConflictType.NO_CONFLICT.value
    recommended_action: str = Field(
        ...,
        description="'add_new', 'keep_existing', 'supersede_existing', 'needs_review', or 'reject'."
    )
    decision_reason: str
    is_explicit_update: bool = False
    evidence_quote: Optional[str] = None
    transition_detected: Optional[str] = None
    confidence_score: float = 1.0


class RawExtractedClaim(BaseModel):
    """Raw claim extracted by LLM before verification."""

    subject: str = Field(..., min_length=1)
    predicate: str = Field(..., min_length=1)
    object_value: str = Field(..., min_length=1)
    source_quote: str = Field(..., min_length=1)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None


class ExtractionResponse(BaseModel):
    """Structured response envelope expected from the extraction prompt."""

    claims: List[RawExtractedClaim] = Field(default_factory=list)


class ExtractionValidationIssue(BaseModel):
    """Details of an extracted claim that failed validation or verification."""

    claim_data: Dict[str, Any] = Field(default_factory=dict)
    reason: str


class ExtractionResult(BaseModel):
    """Overall outcome of a memory extraction operation."""

    valid_claims: List[MemoryClaim] = Field(default_factory=list)
    rejected_claims: List[ExtractionValidationIssue] = Field(default_factory=list)
    raw_model_output: str = ""
    success: bool = True
    error_message: Optional[str] = None


class OllamaConfig(BaseModel):
    """Configuration settings for local Ollama inference."""

    host: str = Field(
        default="http://localhost:11434",
        description="Base URL for the local Ollama instance.",
    )
    model: str = Field(
        default="qwen3.5:0.8b",
        description="Target model identifier in Ollama.",
    )
    timeout_seconds: float = Field(
        default=30.0,
        description="HTTP request timeout in seconds.",
    )

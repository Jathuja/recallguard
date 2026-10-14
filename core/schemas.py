"""Core data models and schemas for RecallGuard."""

from datetime import datetime, timezone
from typing import Any, Dict, Optional
import uuid
from pydantic import BaseModel, Field, field_validator


class MemoryClaim(BaseModel):
    """Represents an atomic memory claim with temporal validity and reliability attributes.
    
    A claim expresses a subject-predicate-object assertion extracted from agent
    interactions, tagged with temporal bounds and confidence ratings.
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
    is_active: bool = Field(
        default=True,
        description="Whether this claim is currently active and eligible for retrieval.",
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

    def is_expired(self, as_of: Optional[datetime] = None) -> bool:
        """Check if this claim is expired as of a given timestamp."""
        check_time = as_of or datetime.now(timezone.utc)
        if self.valid_until is not None:
            # Ensure check_time is timezone-aware if valid_until is timezone-aware
            if self.valid_until.tzinfo is not None and check_time.tzinfo is None:
                check_time = check_time.replace(tzinfo=timezone.utc)
            elif self.valid_until.tzinfo is None and check_time.tzinfo is not None:
                check_time = check_time.astimezone(timezone.utc).replace(tzinfo=None)
            return check_time > self.valid_until
        return False


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

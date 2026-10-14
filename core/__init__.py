"""RecallGuard core package."""

from core.conflict_detector import evaluate_claim_conflict
from core.database import DatabaseManager
from core.evaluator import ReliabilityEvaluator
from core.llm import OllamaClient
from core.memory_service import MemoryService
from core.schemas import (
    ClaimStatus,
    ConflictResolutionDecision,
    ConflictType,
    ExtractionResponse,
    ExtractionResult,
    ExtractionValidationIssue,
    MemoryClaim,
    OllamaConfig,
    RawExtractedClaim,
)

__all__ = [
    "ClaimStatus",
    "ConflictResolutionDecision",
    "ConflictType",
    "DatabaseManager",
    "ExtractionResponse",
    "ExtractionResult",
    "ExtractionValidationIssue",
    "MemoryClaim",
    "MemoryService",
    "OllamaClient",
    "OllamaConfig",
    "RawExtractedClaim",
    "ReliabilityEvaluator",
    "evaluate_claim_conflict",
]

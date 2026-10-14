"""RecallGuard core package."""

from core.database import DatabaseManager
from core.evaluator import ReliabilityEvaluator
from core.llm import OllamaClient
from core.memory_service import MemoryService
from core.schemas import (
    ExtractionResponse,
    ExtractionResult,
    ExtractionValidationIssue,
    MemoryClaim,
    OllamaConfig,
    RawExtractedClaim,
)

__all__ = [
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
]

"""RecallGuard core package."""

from core.conflict_detector import evaluate_claim_conflict
from core.database import DatabaseManager
from core.evaluator import (
    EvalDataset,
    EvalSample,
    EvaluationReport,
    NaiveMemoryBaseline,
    PerSampleResult,
    ReliabilityEvaluator,
    _conflict_in_db,
    build_synthetic_dataset,
)
from core.llm import OllamaClient
from core.memory_service import MemoryService
from core.retrieval_service import (
    RetrievalConfig,
    RetrievalResult,
    RetrievalService,
    _clean_grounded_answer,
)
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
    # Schemas
    "ClaimStatus",
    "ConflictResolutionDecision",
    "ConflictType",
    "ExtractionResponse",
    "ExtractionResult",
    "ExtractionValidationIssue",
    "MemoryClaim",
    "OllamaConfig",
    "RawExtractedClaim",
    # Database
    "DatabaseManager",
    # LLM
    "OllamaClient",
    # Services
    "MemoryService",
    "evaluate_claim_conflict",
    # Feature 04 – Retrieval
    "_clean_grounded_answer",
    "RetrievalConfig",
    "RetrievalResult",
    "RetrievalService",
    # Feature 04 – Evaluator
    "_conflict_in_db",
    "EvalDataset",
    "EvalSample",
    "EvaluationReport",
    "NaiveMemoryBaseline",
    "PerSampleResult",
    "ReliabilityEvaluator",
    "build_synthetic_dataset",
]


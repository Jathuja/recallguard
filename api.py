"""FastAPI Local Agent Integration API for RecallGuard (Feature 05).

Provides HTTP endpoints for local AI agents to:
- Retrieve evidence-grounded, active, non-expired memory claims.
- Ask questions and receive grounded answers with exact source-quote citations.
- Check system health (API, SQLite database, and local Ollama daemon).

Security & Deployment:
- Bound to 127.0.0.1 by default (never exposed publicly or across networks).
- Zero cloud API calls; all inference is local via Ollama with qwen3.5:0.8b.
- Non-destructive endpoints only (read-only retrieval and Q&A).
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from core.database import DatabaseManager
from core.llm import OllamaClient
from core.retrieval_service import RetrievalConfig, RetrievalResult, RetrievalService
from core.schemas import MemoryClaim


# ---------------------------------------------------------------------------
# Pydantic Request / Response Models
# ---------------------------------------------------------------------------

class ClaimResponse(BaseModel):
    """Public representation of an active memory claim."""

    id: str
    subject: str
    predicate: str
    object_value: str
    confidence: float
    source_quote: Optional[str] = None
    source_session_id: Optional[str] = None
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    created_at: datetime
    status: str
    is_active: bool
    superseded_by: Optional[str] = None
    supersedes_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_claim(cls, claim: MemoryClaim) -> ClaimResponse:
        return cls(
            id=claim.id,
            subject=claim.subject,
            predicate=claim.predicate,
            object_value=claim.object_value,
            confidence=claim.confidence,
            source_quote=claim.source_quote,
            source_session_id=claim.source_session_id,
            valid_from=claim.valid_from,
            valid_until=claim.valid_until,
            created_at=claim.created_at,
            status=claim.status,
            is_active=claim.is_active,
            superseded_by=claim.superseded_by,
            supersedes_id=claim.supersedes_id,
            metadata=claim.metadata or {},
        )


class RetrieveRequest(BaseModel):
    """Request payload for POST /v1/memories/retrieve."""

    query: str = Field(..., min_length=1, description="Question or keyword query to search stored memories for.")
    max_results: int = Field(default=10, ge=1, le=50, description="Maximum number of active claims to retrieve.")
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum confidence threshold.")
    include_needs_review: bool = Field(default=False, description="Whether to include unresolved conflicting claims.")

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Query must not be empty or whitespace only.")
        return v.strip()


class RetrieveResponse(BaseModel):
    """Response payload for POST /v1/memories/retrieve."""

    query: str
    claims: List[ClaimResponse]
    total_retrieved: int
    filtered_out_count: int
    confidence: float
    abstained: bool
    abstention_reason: str


class AskRequest(BaseModel):
    """Request payload for POST /v1/ask."""

    query: str = Field(..., min_length=1, description="Question about stored agent memories.")
    max_results: int = Field(default=10, ge=1, le=50, description="Maximum evidence claims to retrieve.")
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum confidence threshold.")

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Query must not be empty or whitespace only.")
        return v.strip()


class AskResponse(BaseModel):
    """Response payload for POST /v1/ask."""

    query: str
    answer: str
    abstained: bool
    abstention_reason: str
    evidence_ids: List[str]
    evidence_quotes: List[str]
    retrieved_claims: List[ClaimResponse]
    confidence: float
    filtered_out_count: int


class DatabaseHealth(BaseModel):
    connected: bool
    path: str
    total_claims: int
    active_claims: int
    error: Optional[str] = None


class OllamaHealth(BaseModel):
    connected: bool
    host: str
    model: str
    model_available: bool
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str   # "healthy", "degraded", "unhealthy"
    api: str      # "online"
    database: DatabaseHealth
    ollama: OllamaHealth
    timestamp: datetime


# ---------------------------------------------------------------------------
# Dependency Providers (support test injection via app.dependency_overrides)
# ---------------------------------------------------------------------------

def get_db() -> DatabaseManager:
    """Provide the default SQLite DatabaseManager."""
    db_path = os.getenv("RECALLGUARD_DB_PATH", "data/recallguard.db")
    db = DatabaseManager(db_path=db_path)
    db.init_db()
    return db


def get_llm() -> OllamaClient:
    """Provide the default OllamaClient."""
    return OllamaClient()


# ---------------------------------------------------------------------------
# FastAPI Application Factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RecallGuard Agent Integration API",
    description="Local Temporal Memory Reliability Firewall for AI Agents (Feature 05)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Restrict CORS to localhost only (local agent safety)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://127.0.0.1",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Info"])
def root_info() -> Dict[str, str]:
    """Root metadata endpoint."""
    return {
        "service": "RecallGuard Agent Integration API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check(
    db: DatabaseManager = Depends(get_db),
    llm: OllamaClient = Depends(get_llm),
) -> HealthResponse:
    """Report API, database, and Ollama availability.

    Returns:
    - status="healthy" if both SQLite and local Ollama (with model) are operational.
    - status="degraded" if SQLite is online but Ollama is unavailable (retrieval works, generation disabled).
    - status="unhealthy" if SQLite is unreachable.
    """
    # 1. Database connectivity check
    db_connected = False
    total_claims = 0
    active_claims = 0
    db_err = None
    try:
        counts = db.count_claims()
        if isinstance(counts, dict):
            total_claims = counts.get("total", 0)
            active_claims = counts.get("active", 0)
        else:
            total_claims = int(counts)
            active_claims = len(db.list_claims(active_only=True))
        db_connected = True
    except Exception as exc:
        db_err = str(exc)

    # 2. Ollama daemon and model check
    ollama_connected = False
    model_available = False
    ollama_err = None
    try:
        ollama_connected = llm.check_connection()
        if ollama_connected:
            model_available = llm.is_model_available()
    except Exception as exc:
        ollama_err = str(exc)

    if db_connected and ollama_connected and model_available:
        overall_status = "healthy"
    elif db_connected:
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"

    return HealthResponse(
        status=overall_status,
        api="online",
        database=DatabaseHealth(
            connected=db_connected,
            path=str(db.db_path),
            total_claims=total_claims,
            active_claims=active_claims,
            error=db_err,
        ),
        ollama=OllamaHealth(
            connected=ollama_connected,
            host=llm.host,
            model=llm.model,
            model_available=model_available,
            error=ollama_err,
        ),
        timestamp=datetime.now(timezone.utc),
    )


@app.post("/v1/memories/retrieve", response_model=RetrieveResponse, tags=["Retrieval"])
def retrieve_memories(
    req: RetrieveRequest,
    db: DatabaseManager = Depends(get_db),
    llm: OllamaClient = Depends(get_llm),
) -> RetrieveResponse:
    """Retrieve relevant active, non-expired, non-superseded memory claims.

    Guarantees:
    - Superseded and expired claims are automatically filtered out.
    - Claims flagged 'needs_review' are excluded unless explicitly requested.
    - Claims are ranked deterministically by keyword overlap, confidence, and recency.
    - Never invents citations or returns ungrounded claims.
    """
    cfg = RetrievalConfig(
        max_results=req.max_results,
        min_confidence=req.min_confidence,
        include_needs_review=req.include_needs_review,
    )
    svc = RetrievalService(db_manager=db, llm_client=llm, config=cfg)
    rr: RetrievalResult = svc.retrieve(req.query)

    return RetrieveResponse(
        query=rr.query,
        claims=[ClaimResponse.from_claim(c) for c in rr.retrieved_claims],
        total_retrieved=len(rr.retrieved_claims),
        filtered_out_count=rr.filtered_out_count,
        confidence=rr.confidence,
        abstained=rr.abstained,
        abstention_reason=rr.abstention_reason,
    )


@app.post("/v1/ask", response_model=AskResponse, tags=["Q&A"])
def ask_question(
    req: AskRequest,
    db: DatabaseManager = Depends(get_db),
    llm: OllamaClient = Depends(get_llm),
) -> AskResponse:
    """Return an evidence-grounded answer citing exact source quotes.

    Guarantees:
    - Retrieves only active, valid claims.
    - Prompts local Ollama to synthesize an answer using ONLY the retrieved evidence.
    - Returns cited evidence IDs (e.g., [E1]) and exact verbatim source quotes.
    - When evidence is insufficient or absent, safely abstains with abstained=True
      and a clear abstention_reason.
    - Never exposes superseded claims as current facts.
    """
    cfg = RetrievalConfig(
        max_results=req.max_results,
        min_confidence=req.min_confidence,
    )
    svc = RetrievalService(db_manager=db, llm_client=llm, config=cfg)
    rr: RetrievalResult = svc.answer(req.query)

    return AskResponse(
        query=rr.query,
        answer=rr.answer,
        abstained=rr.abstained,
        abstention_reason=rr.abstention_reason,
        evidence_ids=rr.evidence_ids,
        evidence_quotes=rr.evidence_quotes,
        retrieved_claims=[ClaimResponse.from_claim(c) for c in rr.retrieved_claims],
        confidence=rr.confidence,
        filtered_out_count=rr.filtered_out_count,
    )


# ---------------------------------------------------------------------------
# Entrypoint for local execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("api:app", host=host, port=port, reload=False)

"""Automated tests for RecallGuard Local Agent Integration API (Feature 05).

Covers:
- Root metadata endpoint
- GET /health: reporting API, SQLite database, and local Ollama status
- POST /v1/memories/retrieve: successful retrieval, status/staleness filtering, confidence, pagination
- POST /v1/ask: grounded answers with citations, missing evidence, model abstention, unavailable Ollama
- Malformed request handling and HTTP 422 validation
- Temporal migration query consistency (Node.js 20 -> Go 1.22)
- Zero reliance on live Ollama or production database (all tests isolated & mocked).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Generator
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import pytest

from api import app, get_db, get_llm
from core.database import DatabaseManager
from core.llm import OllamaClient
from core.schemas import ClaimStatus, MemoryClaim


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def test_db() -> DatabaseManager:
    """In-memory database for isolated API tests."""
    db = DatabaseManager(db_path=":memory:")
    db.init_db()
    return db


@pytest.fixture()
def mock_llm() -> MagicMock:
    """Mock OllamaClient to avoid requiring a live Ollama server."""
    llm = MagicMock(spec=OllamaClient)
    llm.host = "http://localhost:11434"
    llm.model = "qwen3.5:0.8b"
    llm.check_connection.return_value = True
    llm.is_model_available.return_value = True
    llm.generate.return_value = "The user prefers Python for backend development [E1]."
    return llm


@pytest.fixture()
def client(test_db: DatabaseManager, mock_llm: MagicMock) -> Generator[TestClient, None, None]:
    """TestClient with dependency overrides for database and LLM."""
    app.dependency_overrides[get_db] = lambda: test_db
    app.dependency_overrides[get_llm] = lambda: mock_llm
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture()
def populated_db(test_db: DatabaseManager) -> DatabaseManager:
    """Seed test database with active, superseded, expired, and needs_review claims."""
    now = datetime.now(timezone.utc)
    one_day_ago = now - timedelta(days=1)
    two_days_ago = now - timedelta(days=2)

    # Active claim 1: Python preference
    c1 = MemoryClaim(
        subject="user",
        predicate="preferred_language",
        object_value="Python",
        confidence=0.95,
        source_quote="I prefer Python for all backend services.",
        status=ClaimStatus.ACTIVE.value,
        is_active=True,
        created_at=one_day_ago,
    )

    # Active claim 2: Current backend framework (Go 1.22)
    c2 = MemoryClaim(
        subject="backend_service",
        predicate="framework",
        object_value="Go 1.22",
        confidence=0.95,
        source_quote="We decided to migrate the backend service from Node.js 20 to Go 1.22.",
        status=ClaimStatus.ACTIVE.value,
        is_active=True,
        created_at=one_day_ago,
    )

    # Superseded claim: Old backend framework (Node.js 20)
    c_stale = MemoryClaim(
        subject="backend_service",
        predicate="framework",
        object_value="Node.js 20",
        confidence=0.90,
        source_quote="Our backend service runs on Node.js 20.",
        status=ClaimStatus.SUPERSEDED.value,
        is_active=False,
        superseded_by=c2.id,
        created_at=two_days_ago,
    )

    # Expired claim: temporary token
    c_expired = MemoryClaim(
        subject="api_auth",
        predicate="temp_token",
        object_value="xyz-old-token",
        confidence=0.85,
        source_quote="The temporary access token is xyz-old-token.",
        status=ClaimStatus.ACTIVE.value,
        is_active=True,
        valid_until=two_days_ago,
        created_at=two_days_ago,
    )

    # Conflicting / needs_review claim
    c_review = MemoryClaim(
        subject="project",
        predicate="status",
        object_value="delayed",
        confidence=0.70,
        source_quote="The project might be delayed.",
        status=ClaimStatus.NEEDS_REVIEW.value,
        is_active=True,
        created_at=one_day_ago,
    )

    for c in [c1, c2, c_stale, c_expired, c_review]:
        test_db.insert_claim(c)

    return test_db


# ---------------------------------------------------------------------------
# Root and OpenAPI Tests
# ---------------------------------------------------------------------------

class TestApiRootAndDocs:
    def test_root_endpoint_returns_info(self, client: TestClient) -> None:
        res = client.get("/")
        assert res.status_code == 200
        data = res.json()
        assert data["service"] == "RecallGuard Agent Integration API"
        assert "version" in data
        assert data["docs_url"] == "/docs"
        assert data["health_url"] == "/health"

    def test_openapi_json_accessible(self, client: TestClient) -> None:
        res = client.get("/openapi.json")
        assert res.status_code == 200
        assert res.json()["info"]["title"] == "RecallGuard Agent Integration API"


# ---------------------------------------------------------------------------
# Health Endpoint Tests
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_all_healthy(self, client: TestClient, populated_db: DatabaseManager) -> None:
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert data["api"] == "online"
        assert data["database"]["connected"] is True
        assert data["database"]["total_claims"] == 5
        assert data["database"]["active_claims"] == 3
        assert data["ollama"]["connected"] is True
        assert data["ollama"]["model"] == "qwen3.5:0.8b"
        assert data["ollama"]["model_available"] is True

    def test_health_ollama_disconnected_returns_degraded(
        self, client: TestClient, mock_llm: MagicMock
    ) -> None:
        mock_llm.check_connection.return_value = False
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "degraded"
        assert data["ollama"]["connected"] is False

    def test_health_ollama_model_missing_returns_degraded(
        self, client: TestClient, mock_llm: MagicMock
    ) -> None:
        mock_llm.check_connection.return_value = True
        mock_llm.is_model_available.return_value = False
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "degraded"
        assert data["ollama"]["model_available"] is False

    def test_health_database_error_returns_unhealthy(
        self, client: TestClient, test_db: DatabaseManager
    ) -> None:
        broken_db = MagicMock(spec=DatabaseManager)
        broken_db.db_path = ":memory:"
        broken_db.count_claims.side_effect = RuntimeError("Disk I/O error")
        app.dependency_overrides[get_db] = lambda: broken_db

        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "unhealthy"
        assert data["database"]["connected"] is False
        assert "Disk I/O error" in data["database"]["error"]


# ---------------------------------------------------------------------------
# POST /v1/memories/retrieve Tests
# ---------------------------------------------------------------------------

class TestRetrieveEndpoint:
    def test_retrieve_active_claims_success(
        self, client: TestClient, populated_db: DatabaseManager
    ) -> None:
        payload = {"query": "What programming language does the user prefer?"}
        res = client.post("/v1/memories/retrieve", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["query"] == payload["query"]
        assert data["abstained"] is False
        assert data["total_retrieved"] >= 1
        claim = data["claims"][0]
        assert claim["subject"] == "user"
        assert claim["predicate"] == "preferred_language"
        assert claim["object_value"] == "Python"
        assert claim["status"] == "active"
        assert "Python" in claim["source_quote"]

    def test_retrieve_filters_out_superseded_and_expired(
        self, client: TestClient, populated_db: DatabaseManager
    ) -> None:
        """Superseded (Node.js 20) and expired claims must never be returned."""
        payload = {"query": "backend framework service Node.js 20 token"}
        res = client.post("/v1/memories/retrieve", json=payload)
        assert res.status_code == 200
        data = res.json()
        # Should only retrieve the active Go 1.22 claim
        retrieved_values = [c["object_value"] for c in data["claims"]]
        assert "Go 1.22" in retrieved_values
        assert "Node.js 20" not in retrieved_values
        assert "xyz-old-token" not in retrieved_values
        # Superseded and expired claims are counted in filtered_out_count
        assert data["filtered_out_count"] >= 1

    def test_retrieve_excludes_needs_review_by_default(
        self, client: TestClient, populated_db: DatabaseManager
    ) -> None:
        payload = {"query": "project status delayed", "include_needs_review": False}
        res = client.post("/v1/memories/retrieve", json=payload)
        assert res.status_code == 200
        data = res.json()
        retrieved_statuses = [c["status"] for c in data["claims"]]
        assert "needs_review" not in retrieved_statuses

    def test_retrieve_includes_needs_review_when_requested(
        self, client: TestClient, populated_db: DatabaseManager
    ) -> None:
        payload = {"query": "project status delayed", "include_needs_review": True}
        res = client.post("/v1/memories/retrieve", json=payload)
        assert res.status_code == 200
        data = res.json()
        retrieved_values = [c["object_value"] for c in data["claims"]]
        assert "delayed" in retrieved_values

    def test_retrieve_missing_evidence_abstains(
        self, client: TestClient, populated_db: DatabaseManager
    ) -> None:
        payload = {"query": "What is the quantum mechanics wavelength forecast?"}
        res = client.post("/v1/memories/retrieve", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["abstained"] is True
        assert data["total_retrieved"] == 0
        assert data["claims"] == []
        assert "No active memory claims" in data["abstention_reason"]

    def test_retrieve_respects_max_results_and_confidence(
        self, client: TestClient, populated_db: DatabaseManager
    ) -> None:
        payload = {
            "query": "user backend service language framework",
            "max_results": 1,
            "min_confidence": 0.90,
        }
        res = client.post("/v1/memories/retrieve", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert len(data["claims"]) <= 1

    def test_retrieve_malformed_empty_query_returns_422(self, client: TestClient) -> None:
        res = client.post("/v1/memories/retrieve", json={"query": ""})
        assert res.status_code == 422

    def test_retrieve_malformed_whitespace_query_returns_422(self, client: TestClient) -> None:
        res = client.post("/v1/memories/retrieve", json={"query": "   \n\t  "})
        assert res.status_code == 422

    def test_retrieve_invalid_confidence_range_returns_422(self, client: TestClient) -> None:
        res = client.post("/v1/memories/retrieve", json={"query": "test", "min_confidence": 1.5})
        assert res.status_code == 422

    def test_retrieve_invalid_max_results_returns_422(self, client: TestClient) -> None:
        res = client.post("/v1/memories/retrieve", json={"query": "test", "max_results": 0})
        assert res.status_code == 422


# ---------------------------------------------------------------------------
# POST /v1/ask Tests
# ---------------------------------------------------------------------------

class TestAskEndpoint:
    def test_ask_grounded_answer_success(
        self, client: TestClient, populated_db: DatabaseManager, mock_llm: MagicMock
    ) -> None:
        mock_llm.generate.return_value = (
            "The user prefers Python for all backend services [E1], quoting: "
            '"I prefer Python for all backend services."'
        )
        payload = {"query": "What programming language does the user prefer?"}
        res = client.post("/v1/ask", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["abstained"] is False
        assert "Python" in data["answer"]
        assert len(data["evidence_ids"]) >= 1
        assert len(data["evidence_quotes"]) >= 1
        assert len(data["retrieved_claims"]) >= 1
        assert data["retrieved_claims"][0]["object_value"] == "Python"

    def test_ask_missing_evidence_safely_abstains(
        self, client: TestClient, populated_db: DatabaseManager
    ) -> None:
        payload = {"query": "What is the corporate office espresso bean recipe?"}
        res = client.post("/v1/ask", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["abstained"] is True
        assert data["answer"] == ""
        assert "No active memory claims" in data["abstention_reason"]

    def test_ask_ollama_unavailable_abstains_safely(
        self, client: TestClient, populated_db: DatabaseManager, mock_llm: MagicMock
    ) -> None:
        mock_llm.generate.side_effect = RuntimeError("Ollama daemon is not running on localhost:11434")
        payload = {"query": "What programming language does the user prefer?"}
        res = client.post("/v1/ask", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["abstained"] is True
        assert data["answer"] == ""
        assert "Ollama inference failed" in data["abstention_reason"]

    def test_ask_empty_llm_response_abstains(
        self, client: TestClient, populated_db: DatabaseManager, mock_llm: MagicMock
    ) -> None:
        mock_llm.generate.return_value = ""
        payload = {"query": "What programming language does the user prefer?"}
        res = client.post("/v1/ask", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["abstained"] is True
        assert data["answer"] == ""
        assert "empty response" in data["abstention_reason"]

    def test_ask_model_initiated_abstention(
        self, client: TestClient, populated_db: DatabaseManager, mock_llm: MagicMock
    ) -> None:
        mock_llm.generate.return_value = "ABSTAIN: Insufficient evidence on this specific question."
        payload = {"query": "What programming language does the user prefer?"}
        res = client.post("/v1/ask", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["abstained"] is True
        assert data["answer"] == ""
        assert "ABSTAIN" in data["abstention_reason"]

    def test_ask_nodejs_migration_produces_clean_go_answer(
        self, client: TestClient, populated_db: DatabaseManager, mock_llm: MagicMock
    ) -> None:
        """Asking about the old Node.js 20 backend yields a clean answer that Go 1.22
        is current, without an unhandled contradictory trailing ABSTAIN block."""
        mock_llm.generate.return_value = (
            "No, our backend service is no longer running Node.js 20; it has been migrated to Go 1.22 [E1]. "
            'The record states: "We decided to migrate the backend service from Node.js 20 to Go 1.22."\n\n'
            "ABSTAIN: The evidence indicates migration to Go 1.22, but does not state if migration is complete."
        )
        payload = {"query": "Is our backend still running Node.js 20?"}
        res = client.post("/v1/ask", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["abstained"] is False
        assert "Go 1.22" in data["answer"]
        assert "ABSTAIN" not in data["answer"]
        assert len(data["evidence_ids"]) >= 1

    def test_ask_malformed_missing_query_returns_422(self, client: TestClient) -> None:
        res = client.post("/v1/ask", json={})
        assert res.status_code == 422

    def test_ask_malformed_empty_query_returns_422(self, client: TestClient) -> None:
        res = client.post("/v1/ask", json={"query": "   "})
        assert res.status_code == 422


# ---------------------------------------------------------------------------
# Feature 05 Bug-Fix API Regression Tests
# ---------------------------------------------------------------------------

class TestApiIrrelevantClaimFiltering:
    """API-level regression: backend framework query must exclude database claims.

    Reproduces Bug 1 at the HTTP API layer.
    """

    @pytest.fixture()
    def mixed_db(self, test_db: DatabaseManager) -> DatabaseManager:
        framework_claim = MemoryClaim(
            subject="backend_service",
            predicate="framework",
            object_value="Go 1.22",
            confidence=0.95,
            source_quote="We have migrated the backend service from Node.js 20 to Go 1.22.",
            status=ClaimStatus.ACTIVE.value,
            is_active=True,
        )
        database_claim = MemoryClaim(
            subject="backend_system",
            predicate="database_changes",
            object_value="PostgreSQL 16",
            confidence=0.90,
            source_quote="Yes, we are deprecating MongoDB and standardizing on PostgreSQL 16.",
            status=ClaimStatus.ACTIVE.value,
            is_active=True,
        )
        test_db.insert_claim(framework_claim)
        test_db.insert_claim(database_claim)
        return test_db

    def test_retrieve_framework_query_excludes_database_claim(
        self, client: TestClient, mixed_db: DatabaseManager
    ) -> None:
        """POST /v1/memories/retrieve: framework question must not return database claims."""
        payload = {"query": "What backend framework does my project currently use?"}
        res = client.post("/v1/memories/retrieve", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["abstained"] is False
        object_values = [c["object_value"] for c in data["claims"]]
        predicates    = [c["predicate"] for c in data["claims"]]
        assert "Go 1.22" in object_values, "Framework claim must be returned"
        assert "PostgreSQL 16" not in object_values, "Database claim must NOT be returned"
        assert "database_changes" not in predicates, "database_changes must be excluded"

    def test_ask_framework_query_excludes_database_claim(
        self, client: TestClient, mixed_db: DatabaseManager, mock_llm: MagicMock
    ) -> None:
        """POST /v1/ask: framework question answer must be grounded only in framework evidence."""
        mock_llm.generate.return_value = (
            "The backend service uses Go 1.22 [E1]. "
            '"We have migrated the backend service from Node.js 20 to Go 1.22."'
        )
        payload = {"query": "What backend framework does my project currently use?"}
        res = client.post("/v1/ask", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["abstained"] is False
        assert "Go 1.22" in data["answer"]
        # PostgreSQL must not appear in retrieved claims
        claim_predicates = [c["predicate"] for c in data["retrieved_claims"]]
        assert "database_changes" not in claim_predicates


class TestApiOverlappingEvidenceDeduplication:
    """API-level regression: overlapping Go 1.22 claims must not both be returned.

    Reproduces Bug 2 at the HTTP API layer.
    """

    @pytest.fixture()
    def overlapping_db(self, test_db: DatabaseManager) -> DatabaseManager:
        claim_a = MemoryClaim(
            subject="backend_service",
            predicate="framework",
            object_value="Go 1.22",
            confidence=0.95,
            source_quote="We decided to migrate the backend service from Node.js 20 to Go 1.22.",
            status=ClaimStatus.ACTIVE.value,
            is_active=True,
        )
        claim_b = MemoryClaim(
            subject="backend_system",
            predicate="migration_target",
            object_value="Go 1.22",
            confidence=0.88,
            source_quote="We officially decided to migrate our backend from Node.js to Go 1.22 starting next month.",
            status=ClaimStatus.ACTIVE.value,
            is_active=True,
        )
        test_db.insert_claim(claim_a)
        test_db.insert_claim(claim_b)
        return test_db

    def test_retrieve_deduplicates_overlapping_go_claims(
        self, client: TestClient, overlapping_db: DatabaseManager
    ) -> None:
        """POST /v1/memories/retrieve must not return two Go 1.22 claims for the same query."""
        payload = {"query": "What backend framework does my project currently use?"}
        res = client.post("/v1/memories/retrieve", json=payload)
        assert res.status_code == 200
        data = res.json()
        go_claims = [c for c in data["claims"] if c["object_value"] == "Go 1.22"]
        assert len(go_claims) <= 1, (
            f"Expected at most 1 Go 1.22 claim; got {len(go_claims)}: "
            f"{[(c['subject'], c['predicate']) for c in go_claims]}"
        )


class TestApiPlannedVsCompletedMigration:
    """API-level regression: planned migration must not be stated as completed.

    Reproduces Bug 3 at the HTTP API layer.
    """

    @pytest.fixture()
    def planned_db(self, test_db: DatabaseManager) -> DatabaseManager:
        planned_claim = MemoryClaim(
            subject="backend_service",
            predicate="framework",
            object_value="Go 1.22",
            confidence=0.88,
            source_quote="We officially decided to migrate our backend from Node.js to Go 1.22 starting next month.",
            status=ClaimStatus.ACTIVE.value,
            is_active=True,
        )
        test_db.insert_claim(planned_claim)
        return test_db

    @pytest.fixture()
    def completed_db(self, test_db: DatabaseManager) -> DatabaseManager:
        completed_claim = MemoryClaim(
            subject="backend_service",
            predicate="framework",
            object_value="Go 1.22",
            confidence=0.95,
            source_quote="We have migrated the backend service from Node.js 20 to Go 1.22.",
            status=ClaimStatus.ACTIVE.value,
            is_active=True,
        )
        test_db.insert_claim(completed_claim)
        return test_db

    def test_retrieve_planned_migration_evidence_block_annotated(
        self, client: TestClient, planned_db: DatabaseManager
    ) -> None:
        """POST /v1/memories/retrieve for a planned-only quote returns exactly one claim."""
        payload = {"query": "What backend framework does my project currently use?"}
        res = client.post("/v1/memories/retrieve", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["abstained"] is False
        # Exactly one Go 1.22 claim retrieved
        go_claims = [c for c in data["claims"] if c["object_value"] == "Go 1.22"]
        assert len(go_claims) == 1

    def test_ask_planned_migration_hallucination_corrected(
        self, client: TestClient, planned_db: DatabaseManager, mock_llm: MagicMock
    ) -> None:
        """POST /v1/ask: when LLM hallucinates a completed migration from a future-tense quote,
        the post-processor must replace the answer with accurate planned-migration language."""
        mock_llm.generate.return_value = (
            "The backend is currently running Go 1.22 [E1]. "
            '"We officially decided to migrate our backend from Node.js to Go 1.22 starting next month."'
        )
        payload = {"query": "What backend framework does my project currently use?"}
        res = client.post("/v1/ask", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["abstained"] is False
        answer_lower = data["answer"].lower()
        assert any(
            kw in answer_lower
            for kw in ["planned", "not yet confirmed", "starting next month"]
        ), f"Expected planned-migration language; got: {data['answer']!r}"

    def test_ask_completed_migration_states_go_current(
        self, client: TestClient, completed_db: DatabaseManager, mock_llm: MagicMock
    ) -> None:
        """POST /v1/ask: past-tense migration quote allows LLM to state Go 1.22 is current."""
        mock_llm.generate.return_value = (
            "The backend service now uses Go 1.22 [E1]. "
            '"We have migrated the backend service from Node.js 20 to Go 1.22."'
        )
        payload = {"query": "What backend framework does my project currently use?"}
        res = client.post("/v1/ask", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["abstained"] is False
        assert "Go 1.22" in data["answer"]

    def test_ask_missing_evidence_safely_abstains_api(
        self, client: TestClient, test_db: DatabaseManager
    ) -> None:
        """POST /v1/ask: unrelated question with empty DB abstains correctly at API layer."""
        payload = {"query": "What is the quantum computing roadmap for fiscal year 2030?"}
        res = client.post("/v1/ask", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["abstained"] is True
        assert data["answer"] == ""
        assert data["abstention_reason"] != ""


class TestApiDecisionVsCompletedMigration:
    """API-level regression: decision to migrate must not be treated as completed.

    Reproduces Issue 1: query 'What backend framework does my project currently use?'
    with quote 'We decided to migrate the backend service from Node.js 20 to Go 1.22.'
    must explain the decision is recorded but completion unconfirmed, and abstain.
    Also tests Issue 2: evidence_quotes must be populated with verbatim source quotes.
    """

    @pytest.fixture()
    def decision_db(self, test_db: DatabaseManager) -> DatabaseManager:
        claim = MemoryClaim(
            subject="backend_service",
            predicate="framework",
            object_value="Go 1.22",
            confidence=0.95,
            source_quote="We decided to migrate the backend service from Node.js 20 to Go 1.22.",
            status=ClaimStatus.ACTIVE.value,
            is_active=True,
        )
        test_db.insert_claim(claim)
        return test_db

    def test_ask_decision_only_abstains_on_current_framework_query(
        self, client: TestClient, decision_db: DatabaseManager
    ) -> None:
        """When question requires knowing current framework, decision-only evidence must abstain."""
        payload = {"query": "What backend framework does my project currently use?"}
        res = client.post("/v1/ask", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["abstained"] is True
        assert data["answer"] == ""
        # Must explain migration decision recorded but completion unconfirmed
        reason_lower = data["abstention_reason"].lower()
        assert "decision" in reason_lower
        assert "unconfirmed" in reason_lower
        # Must not assert current use
        assert "Go 1.22 is currently in use" not in data["answer"]
        # Issue 2: evidence_ids and evidence_quotes must be populated
        assert len(data["evidence_ids"]) == 1
        assert len(data["evidence_quotes"]) == 1
        assert "We decided to migrate" in data["evidence_quotes"][0]

    def test_ask_decision_only_evidence_quotes_always_populated(
        self, client: TestClient, decision_db: DatabaseManager, mock_llm: MagicMock
    ) -> None:
        """Issue 2: /v1/ask returns populated evidence_quotes with verbatim source quotes."""
        mock_llm.generate.return_value = (
            "A decision was made to migrate to Go 1.22 [E1]. "
            '"We decided to migrate the backend service from Node.js 20 to Go 1.22."'
        )
        payload = {"query": "What did the team decide about the backend service framework?"}
        res = client.post("/v1/ask", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["abstained"] is False
        assert len(data["evidence_ids"]) >= 1
        assert len(data["evidence_quotes"]) >= 1
        assert data["evidence_quotes"][0] == "We decided to migrate the backend service from Node.js 20 to Go 1.22."

    def test_ask_evidence_quotes_populated_when_llm_omits_citation(
        self, client: TestClient, decision_db: DatabaseManager, mock_llm: MagicMock
    ) -> None:
        """Issue 2: even if LLM generates answer without [E1] markers, evidence_quotes is populated."""
        mock_llm.generate.return_value = (
            "The team decided on migrating the service to Go 1.22, but completion is not verified."
        )
        payload = {"query": "What did the team decide about the backend service framework?"}
        res = client.post("/v1/ask", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert len(data["evidence_quotes"]) >= 1
        assert "We decided to migrate" in data["evidence_quotes"][0]

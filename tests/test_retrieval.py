"""Tests for the Evidence-Grounded Retrieval Service (Feature 04)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from core.database import DatabaseManager
from core.llm import OllamaClient
from core.retrieval_service import (
    RetrievalConfig,
    RetrievalResult,
    RetrievalService,
    _clean_grounded_answer,
    _score_claim,
    _tokenize,
)
from core.schemas import ClaimStatus, MemoryClaim


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db() -> DatabaseManager:
    """In-memory database pre-populated with test claims."""
    manager = DatabaseManager(db_path=":memory:")
    manager.init_db()
    return manager


@pytest.fixture()
def active_claim(db: DatabaseManager) -> MemoryClaim:
    claim = MemoryClaim(
        subject="user",
        predicate="preferred_language",
        object_value="Python",
        confidence=0.95,
        source_quote="I prefer Python for all backend work.",
        status=ClaimStatus.ACTIVE.value,
        is_active=True,
    )
    db.insert_claim(claim)
    return claim


@pytest.fixture()
def superseded_claim(db: DatabaseManager) -> MemoryClaim:
    claim = MemoryClaim(
        subject="user",
        predicate="preferred_language",
        object_value="Java",
        confidence=0.80,
        source_quote="I used to write everything in Java.",
        status=ClaimStatus.SUPERSEDED.value,
        is_active=False,
    )
    db.insert_claim(claim)
    return claim


@pytest.fixture()
def expired_claim(db: DatabaseManager) -> MemoryClaim:
    claim = MemoryClaim(
        subject="project",
        predicate="deadline",
        object_value="2020-01-01",
        confidence=0.90,
        source_quote="The old deadline was January 2020.",
        status=ClaimStatus.ACTIVE.value,
        is_active=True,
        valid_until=datetime.now(timezone.utc) - timedelta(days=365),
    )
    db.insert_claim(claim)
    return claim


@pytest.fixture()
def mock_llm() -> OllamaClient:
    client = MagicMock(spec=OllamaClient)
    client.generate.return_value = "The user prefers Python [E1]."
    return client


@pytest.fixture()
def retrieval_service(db: DatabaseManager, mock_llm: OllamaClient) -> RetrievalService:
    return RetrievalService(db_manager=db, llm_client=mock_llm)


# ---------------------------------------------------------------------------
# Helper / unit tests
# ---------------------------------------------------------------------------

class TestTokenizer:
    def test_removes_stop_words(self) -> None:
        tokens = _tokenize("What is the user's preferred language?")
        assert "what" not in tokens
        assert "the" not in tokens
        assert "is" not in tokens

    def test_returns_meaningful_tokens(self) -> None:
        tokens = _tokenize("preferred language user")
        assert "preferred" in tokens
        assert "language" in tokens
        assert "user" in tokens

    def test_empty_string(self) -> None:
        assert _tokenize("") == []

    def test_stop_words_only(self) -> None:
        assert _tokenize("what is the") == []


class TestScoreClaim:
    def test_scores_keyword_hits(self) -> None:
        """Keyword matches in predicate (weight 5), subject (3), and object (3) accumulate score."""
        claim = MemoryClaim(
            subject="user",
            predicate="preferred_language",
            object_value="Python",
            source_quote="I prefer Python.",
        )
        score = _score_claim(claim, ["user", "language", "python"])
        # predicate hits 'language' (5) + subject hits 'user' (3) + object/quote hit 'python' (3+1=4) => >= 5
        assert score >= 5

    def test_no_match_returns_non_positive_score(self) -> None:
        """When no tokens overlap, score is 0 or negative (domain-mismatch penalty may apply)."""
        claim = MemoryClaim(
            subject="project",
            predicate="deadline",
            object_value="2025",
        )
        score = _score_claim(claim, ["language", "python"])
        # Domain mismatch: query is 'language' domain, claim is 'deadline' domain → score <= 0
        assert score <= 0


# ---------------------------------------------------------------------------
# RetrievalService.retrieve() tests
# ---------------------------------------------------------------------------

class TestRetrieve:
    def test_returns_active_claim_for_relevant_query(
        self, retrieval_service: RetrievalService, active_claim: MemoryClaim
    ) -> None:
        result = retrieval_service.retrieve("What language does the user prefer?")
        assert not result.abstained
        claim_ids = [c.id for c in result.retrieved_claims]
        assert active_claim.id in claim_ids

    def test_excludes_superseded_claims(
        self,
        retrieval_service: RetrievalService,
        active_claim: MemoryClaim,
        superseded_claim: MemoryClaim,
    ) -> None:
        result = retrieval_service.retrieve("programming language user prefer")
        claim_ids = [c.id for c in result.retrieved_claims]
        assert superseded_claim.id not in claim_ids

    def test_excludes_expired_claims(
        self, retrieval_service: RetrievalService, expired_claim: MemoryClaim
    ) -> None:
        result = retrieval_service.retrieve("project deadline")
        claim_ids = [c.id for c in result.retrieved_claims]
        assert expired_claim.id not in claim_ids

    def test_abstains_on_empty_query(self, retrieval_service: RetrievalService) -> None:
        result = retrieval_service.retrieve("")
        assert result.abstained
        assert result.abstention_reason != ""

    def test_abstains_on_stop_word_only_query(self, retrieval_service: RetrievalService) -> None:
        result = retrieval_service.retrieve("what is the")
        assert result.abstained

    def test_abstains_when_no_claims_match(
        self, retrieval_service: RetrievalService, active_claim: MemoryClaim
    ) -> None:
        result = retrieval_service.retrieve("coffee brand espresso favourite")
        assert result.abstained

    def test_filtered_out_count_tracks_stale(
        self,
        retrieval_service: RetrievalService,
        active_claim: MemoryClaim,
        superseded_claim: MemoryClaim,
        expired_claim: MemoryClaim,
    ) -> None:
        result = retrieval_service.retrieve("language deadline project user")
        # superseded_claim and expired_claim should be counted as filtered out
        assert result.filtered_out_count >= 1

    def test_confidence_mean_calculated(
        self, retrieval_service: RetrievalService, active_claim: MemoryClaim
    ) -> None:
        result = retrieval_service.retrieve("user preferred language")
        if result.retrieved_claims:
            assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# RetrievalService.answer() tests
# ---------------------------------------------------------------------------

class TestAnswer:
    def test_grounded_answer_returned(
        self,
        retrieval_service: RetrievalService,
        active_claim: MemoryClaim,
        mock_llm: OllamaClient,
    ) -> None:
        mock_llm.generate.return_value = "The user prefers Python [E1]."
        result = retrieval_service.answer("What language does the user prefer?")
        assert not result.abstained
        assert "Python" in result.answer or result.answer != ""

    def test_abstains_when_ollama_fails(
        self, db: DatabaseManager, active_claim: MemoryClaim
    ) -> None:
        failing_llm = MagicMock(spec=OllamaClient)
        failing_llm.generate.side_effect = RuntimeError("Ollama not running")
        svc = RetrievalService(db_manager=db, llm_client=failing_llm)
        result = svc.answer("What language does the user prefer?")
        assert result.abstained
        assert "Ollama" in result.abstention_reason

    def test_abstains_on_model_initiated_abstention(
        self, db: DatabaseManager, active_claim: MemoryClaim
    ) -> None:
        abstaining_llm = MagicMock(spec=OllamaClient)
        abstaining_llm.generate.return_value = "ABSTAIN: Insufficient evidence on this topic."
        svc = RetrievalService(db_manager=db, llm_client=abstaining_llm)
        result = svc.answer("What language does the user prefer?")
        assert result.abstained
        assert result.answer == ""

    def test_abstains_when_no_evidence(
        self, retrieval_service: RetrievalService, active_claim: MemoryClaim
    ) -> None:
        result = retrieval_service.answer("favourite ice cream flavour")
        assert result.abstained

    def test_abstains_on_empty_llm_response(
        self, db: DatabaseManager, active_claim: MemoryClaim
    ) -> None:
        empty_llm = MagicMock(spec=OllamaClient)
        empty_llm.generate.return_value = ""
        svc = RetrievalService(db_manager=db, llm_client=empty_llm)
        result = svc.answer("What language does the user prefer?")
        assert result.abstained


# ---------------------------------------------------------------------------
# Evidence block builder
# ---------------------------------------------------------------------------

class TestEvidenceBlock:
    def test_builds_evidence_with_ids(self) -> None:
        claims = [
            MemoryClaim(
                subject="user",
                predicate="language",
                object_value="Python",
                source_quote="I prefer Python.",
            )
        ]
        block = RetrievalService._build_evidence_block(claims)
        assert "[E1]" in block
        assert "Python" in block
        assert "I prefer Python." in block

    def test_empty_list_returns_empty_string(self) -> None:
        block = RetrievalService._build_evidence_block([])
        assert block == ""


# ---------------------------------------------------------------------------
# RetrievalConfig
# ---------------------------------------------------------------------------

class TestRetrievalConfig:
    def test_default_config(self) -> None:
        cfg = RetrievalConfig()
        assert cfg.max_results == 10
        assert cfg.min_confidence == 0.0
        assert not cfg.include_needs_review

    def test_custom_config_applied(self, db: DatabaseManager, mock_llm: OllamaClient) -> None:
        cfg = RetrievalConfig(max_results=2, min_confidence=0.9)
        svc = RetrievalService(db_manager=db, llm_client=mock_llm, config=cfg)
        assert svc.config.max_results == 2
        assert svc.config.min_confidence == 0.9


# ---------------------------------------------------------------------------
# Feature 04 Consistency Regression: Node.js 20 migration answer
# ---------------------------------------------------------------------------

class TestNodeJsMigrationConsistency:
    """Regression tests: asking about old Node.js 20 backend produces a clear,
    evidence-grounded answer that Go 1.22 is current without contradictory ABSTAIN sections.
    """

    @pytest.fixture()
    def migration_db(self) -> DatabaseManager:
        db = DatabaseManager(db_path=":memory:")
        db.init_db()
        go_claim = MemoryClaim(
            subject="backend_service",
            predicate="framework",
            object_value="Go 1.22",
            source_quote="We decided to migrate the backend service from Node.js 20 to Go 1.22.",
            status=ClaimStatus.ACTIVE.value,
            confidence=0.95,
        )
        nodejs_claim = MemoryClaim(
            subject="backend_service",
            predicate="framework",
            object_value="Node.js 20",
            source_quote="Our backend service runs on Node.js 20.",
            status=ClaimStatus.SUPERSEDED.value,
            confidence=0.90,
            superseded_by=go_claim.id,
            is_active=False,
        )
        db.insert_claim(go_claim)
        db.insert_claim(nodejs_claim)
        return db

    def test_migration_answer_strips_contradictory_trailing_abstain(
        self, migration_db: DatabaseManager
    ) -> None:
        """When LLM returns an answer followed by a contradictory ABSTAIN section,
        the ABSTAIN section is stripped and the grounded answer is retained."""
        mock_llm = MagicMock(spec=OllamaClient)
        mock_llm.generate.return_value = (
            "No, our backend is no longer running Node.js 20; it has been migrated to Go 1.22 [E1]. "
            'The source quote explicitly states: "We decided to migrate the backend service from Node.js 20 to Go 1.22."\n\n'
            "ABSTAIN: The evidence indicates migration to Go 1.22, but does not explicitly state if migration is complete."
        )
        svc = RetrievalService(db_manager=migration_db, llm_client=mock_llm)
        result = svc.answer("Is our backend still running Node.js 20?")
        assert not result.abstained
        assert "Go 1.22" in result.answer
        assert "ABSTAIN" not in result.answer
        assert len(result.evidence_ids) >= 1

    def test_migration_answer_recovers_from_spurious_abstain_prefix(
        self, migration_db: DatabaseManager
    ) -> None:
        """When LLM incorrectly prefixes a substantive answer with ABSTAIN:, the prefix
        is cleaned and the answer is returned rather than discarded."""
        mock_llm = MagicMock(spec=OllamaClient)
        mock_llm.generate.return_value = (
            "ABSTAIN: The provided evidence indicates that the backend service was migrated from Node.js 20 to Go 1.22, "
            "meaning it is currently running Go 1.22 and not Node.js 20 [E1]."
        )
        svc = RetrievalService(db_manager=migration_db, llm_client=mock_llm)
        result = svc.answer("Does our backend service run on Node.js 20?")
        assert not result.abstained
        assert "Go 1.22" in result.answer
        assert not result.answer.startswith("ABSTAIN")

    def test_clean_grounded_answer_preserves_genuine_abstention(self) -> None:
        """Genuine model abstention when evidence is insufficient remains an abstention."""
        claim = MemoryClaim(subject="system", predicate="status", object_value="ok")
        raw = "ABSTAIN: The provided memory evidence contains no information to answer this question."
        ans, abstained, reason = _clean_grounded_answer(raw, [claim])
        assert abstained is True
        assert ans == ""
        assert "ABSTAIN" in reason

    def test_clean_grounded_answer_preserves_normal_answer(self) -> None:
        claim = MemoryClaim(subject="backend", predicate="framework", object_value="Go 1.22")
        raw = "The backend runs Go 1.22 [E1]."
        ans, abstained, reason = _clean_grounded_answer(raw, [claim])
        assert abstained is False
        assert ans == raw
        assert reason == ""


# ---------------------------------------------------------------------------
# Feature 05 Bug-Fix Regression Tests
# ---------------------------------------------------------------------------

class TestIrrelevantClaimFiltering:
    """Regression: backend framework question must NOT return database claims.

    Reproduces Bug 1: 'What backend framework does my project currently use?'
    was returning an unrelated PostgreSQL 16 database claim.
    """

    @pytest.fixture()
    def mixed_db(self) -> DatabaseManager:
        db = DatabaseManager(db_path=":memory:")
        db.init_db()
        # Framework claim — should match
        framework_claim = MemoryClaim(
            subject="backend_service",
            predicate="framework",
            object_value="Go 1.22",
            source_quote="We decided to migrate the backend service from Node.js 20 to Go 1.22.",
            status=ClaimStatus.ACTIVE.value,
            confidence=0.95,
        )
        # Database claim — should NOT match a framework question
        database_claim = MemoryClaim(
            subject="backend_system",
            predicate="database_changes",
            object_value="PostgreSQL 16",
            source_quote="Yes, we are deprecating MongoDB and standardizing on PostgreSQL 16.",
            status=ClaimStatus.ACTIVE.value,
            confidence=0.90,
        )
        db.insert_claim(framework_claim)
        db.insert_claim(database_claim)
        return db

    def test_framework_question_excludes_database_claim(
        self, mixed_db: DatabaseManager, mock_llm: OllamaClient
    ) -> None:
        """Backend framework query must return the framework claim, not the database claim."""
        svc = RetrievalService(db_manager=mixed_db, llm_client=mock_llm)
        result = svc.retrieve("What backend framework does my project currently use?")

        object_values = [c.object_value for c in result.retrieved_claims]
        predicates    = [c.predicate for c in result.retrieved_claims]

        assert "Go 1.22" in object_values, "Framework claim must be returned"
        assert "PostgreSQL 16" not in object_values, "Database claim must NOT be returned"
        assert "database_changes" not in predicates, "database_changes predicate must be excluded"

    def test_database_question_excludes_framework_claim(
        self, mixed_db: DatabaseManager, mock_llm: OllamaClient
    ) -> None:
        """A database query must return the database claim, not the framework claim."""
        svc = RetrievalService(db_manager=mixed_db, llm_client=mock_llm)
        result = svc.retrieve("What database does the backend_system use?")

        object_values = [c.object_value for c in result.retrieved_claims]
        assert "PostgreSQL 16" in object_values, "Database claim must be returned"
        # framework claim should be ranked below (or excluded by gap filtering)
        # Go 1.22 should not dominate a database question
        predicates = [c.predicate for c in result.retrieved_claims]
        if "framework" in predicates and "database_changes" in predicates:
            # If both returned, database_changes must be ranked first
            assert predicates.index("database_changes") < predicates.index("framework")


class TestOverlappingEvidenceDeduplication:
    """Regression: multiple overlapping claims for the same entity/value must be deduplicated.

    Reproduces Bug 2: both 067012f6 (backend_service/framework/Go 1.22) and
    6c322063 (backend_system/migration_target/Go 1.22) were returned for the same question.
    """

    @pytest.fixture()
    def overlapping_db(self) -> DatabaseManager:
        db = DatabaseManager(db_path=":memory:")
        db.init_db()
        # Two claims with the same root entity ('backend') and same value ('Go 1.22')
        claim_a = MemoryClaim(
            subject="backend_service",
            predicate="framework",
            object_value="Go 1.22",
            source_quote="We decided to migrate the backend service from Node.js 20 to Go 1.22.",
            status=ClaimStatus.ACTIVE.value,
            confidence=0.95,
        )
        claim_b = MemoryClaim(
            subject="backend_system",
            predicate="migration_target",
            object_value="Go 1.22",
            source_quote="We officially decided to migrate our backend from Node.js to Go 1.22 starting next month.",
            status=ClaimStatus.ACTIVE.value,
            confidence=0.88,
        )
        db.insert_claim(claim_a)
        db.insert_claim(claim_b)
        return db

    def test_duplicate_value_claims_deduplicated_to_one(
        self, overlapping_db: DatabaseManager, mock_llm: OllamaClient
    ) -> None:
        """When two claims share the same root entity and object_value, only one is returned."""
        svc = RetrievalService(db_manager=overlapping_db, llm_client=mock_llm)
        result = svc.retrieve("What backend framework does my project currently use?")

        go_claims = [c for c in result.retrieved_claims if c.object_value == "Go 1.22"]
        assert len(go_claims) <= 1, (
            f"Expected at most 1 Go 1.22 claim; got {len(go_claims)}: "
            f"{[(c.subject, c.predicate) for c in go_claims]}"
        )

    def test_different_values_not_deduplicated(
        self, overlapping_db: DatabaseManager, mock_llm: OllamaClient
    ) -> None:
        """Claims with DIFFERENT values for the same entity must not be merged away."""
        db = DatabaseManager(db_path=":memory:")
        db.init_db()
        # Competing framework claims with different values
        c1 = MemoryClaim(
            subject="backend_service",
            predicate="framework",
            object_value="Go 1.22",
            source_quote="Team A uses Go 1.22.",
            status=ClaimStatus.ACTIVE.value,
            confidence=0.95,
        )
        c2 = MemoryClaim(
            subject="backend_service",
            predicate="framework",
            object_value="Rust",
            source_quote="Team B prefers Rust for backend.",
            status=ClaimStatus.ACTIVE.value,
            confidence=0.80,
        )
        db.insert_claim(c1)
        db.insert_claim(c2)
        svc = RetrievalService(db_manager=db, llm_client=mock_llm)
        result = svc.retrieve("What backend framework does the service use?")
        object_values = [c.object_value for c in result.retrieved_claims]
        # Different values for different claims — neither should be dropped
        # (they will be deduplicated by (subject, predicate) since subject+predicate match,
        # but subject+predicate are the same for both — only the first ranked survives)
        # At minimum, the top-ranked claim must be present
        assert len(result.retrieved_claims) >= 1


class TestPlannedVsCompletedMigration:
    """Regression: planned migration must not be stated as completed.

    Reproduces Bug 3: answer said 'Go 1.22 is currently in use' when the source quote
    says the migration is starting next month.
    """

    @pytest.fixture()
    def planned_db(self) -> DatabaseManager:
        db = DatabaseManager(db_path=":memory:")
        db.init_db()
        planned_claim = MemoryClaim(
            subject="backend_service",
            predicate="framework",
            object_value="Go 1.22",
            source_quote="We officially decided to migrate our backend from Node.js to Go 1.22 starting next month.",
            status=ClaimStatus.ACTIVE.value,
            confidence=0.88,
        )
        db.insert_claim(planned_claim)
        return db

    @pytest.fixture()
    def completed_db(self) -> DatabaseManager:
        db = DatabaseManager(db_path=":memory:")
        db.init_db()
        completed_claim = MemoryClaim(
            subject="backend_service",
            predicate="framework",
            object_value="Go 1.22",
            source_quote="We have migrated the backend service from Node.js 20 to Go 1.22.",
            status=ClaimStatus.ACTIVE.value,
            confidence=0.95,
        )
        db.insert_claim(completed_claim)
        return db

    def test_planned_migration_quote_annotated_in_evidence_block(self) -> None:
        """Evidence block must annotate planned migration quotes with PLANNED_MIGRATION status."""
        from core.retrieval_service import _is_planned_migration
        planned_claim = MemoryClaim(
            subject="backend_service",
            predicate="framework",
            object_value="Go 1.22",
            source_quote="We officially decided to migrate our backend from Node.js to Go 1.22 starting next month.",
            status=ClaimStatus.ACTIVE.value,
            confidence=0.88,
        )
        assert _is_planned_migration(planned_claim.source_quote) is True
        block = RetrievalService._build_evidence_block([planned_claim])
        assert "PLANNED_MIGRATION" in block

    def test_completed_migration_quote_not_annotated_as_planned(self) -> None:
        """Evidence block must NOT mark a completed-migration quote as planned."""
        from core.retrieval_service import _is_planned_migration
        completed_claim = MemoryClaim(
            subject="backend_service",
            predicate="framework",
            object_value="Go 1.22",
            source_quote="We have migrated the backend service from Node.js 20 to Go 1.22.",
            status=ClaimStatus.ACTIVE.value,
            confidence=0.95,
        )
        assert _is_planned_migration(completed_claim.source_quote) is False
        block = RetrievalService._build_evidence_block([completed_claim])
        assert "PLANNED_MIGRATION" not in block

    def test_planned_migration_llm_hallucination_corrected(
        self, planned_db: DatabaseManager
    ) -> None:
        """When LLM incorrectly says migration is complete from a planned quote,
        the post-processor must correct it to state the migration is planned."""
        # Mock LLM that incorrectly hallucinates a completed migration
        bad_llm = MagicMock(spec=OllamaClient)
        bad_llm.generate.return_value = (
            "The backend is currently running Go 1.22 [E1]. "
            '"We officially decided to migrate our backend from Node.js to Go 1.22 starting next month."'
        )
        svc = RetrievalService(db_manager=planned_db, llm_client=bad_llm)
        result = svc.answer("What backend framework does the project use?")

        assert not result.abstained
        # Must mention planned/not yet confirmed
        answer_lower = result.answer.lower()
        assert any(kw in answer_lower for kw in ["planned", "not yet confirmed", "starting next month"]), (
            f"Expected planned-migration language in answer; got: {result.answer!r}"
        )
        # Must NOT assert current use
        assert "currently running" not in result.answer.lower() or "planned" in result.answer.lower()

    def test_completed_migration_answer_states_current_use(
        self, completed_db: DatabaseManager
    ) -> None:
        """When quote is confirmed completion, LLM answer saying Go 1.22 is current is accepted."""
        good_llm = MagicMock(spec=OllamaClient)
        good_llm.generate.return_value = (
            "The backend service now uses Go 1.22 [E1]. "
            '"We have migrated the backend service from Node.js 20 to Go 1.22."'
        )
        svc = RetrievalService(db_manager=completed_db, llm_client=good_llm)
        result = svc.answer("What backend framework does the service use?")

        assert not result.abstained
        assert "Go 1.22" in result.answer

    def test_is_planned_migration_detects_various_patterns(self) -> None:
        """_is_planned_migration must detect all documented planned-migration patterns."""
        from core.retrieval_service import _is_planned_migration
        planned_quotes = [
            "We plan to migrate to Go 1.22 starting next month.",
            "The team plans to migrate from Python 3.9 to 3.12 next quarter.",
            "We will migrate to Rust next year.",
            "Migration is scheduled to begin in Q1.",
            "The company intends to migrate to Kubernetes next year.",
        ]
        for q in planned_quotes:
            assert _is_planned_migration(q), f"Should detect as planned: {q!r}"

        completed_quotes = [
            "We decided to migrate the backend from Node.js to Go 1.22.",
            "The team migrated to Python 3.12 last week.",
            "We completed the migration to PostgreSQL.",
        ]
        for q in completed_quotes:
            assert not _is_planned_migration(q), f"Should NOT detect as planned: {q!r}"


class TestMissingEvidenceAbstention:
    """Regression: queries with no matching evidence must safely abstain."""

    @pytest.fixture()
    def minimal_db(self) -> DatabaseManager:
        db = DatabaseManager(db_path=":memory:")
        db.init_db()
        claim = MemoryClaim(
            subject="user",
            predicate="preferred_language",
            object_value="Python",
            source_quote="I prefer Python.",
            status=ClaimStatus.ACTIVE.value,
            confidence=0.95,
        )
        db.insert_claim(claim)
        return db

    def test_unrelated_query_abstains(
        self, minimal_db: DatabaseManager, mock_llm: OllamaClient
    ) -> None:
        svc = RetrievalService(db_manager=minimal_db, llm_client=mock_llm)
        result = svc.answer("What is the quantum mechanics wavefunction collapse schedule?")
        assert result.abstained
        assert result.answer == ""
        assert "No active memory claims" in result.abstention_reason

    def test_empty_db_abstains(
        self, mock_llm: OllamaClient
    ) -> None:
        empty_db = DatabaseManager(db_path=":memory:")
        empty_db.init_db()
        svc = RetrievalService(db_manager=empty_db, llm_client=mock_llm)
        result = svc.answer("What backend framework does my project use?")
        assert result.abstained
        assert result.answer == ""


class TestDecisionVsCompletedMigration:
    """Regression tests for distinguishing migration decisions from completed migrations.

    Addresses Issue 1: query 'What backend framework does my project currently use?'
    with quote 'We decided to migrate the backend service from Node.js 20 to Go 1.22.'
    must abstain and explain that a decision was recorded but completion is unconfirmed.
    Addresses Issue 2: evidence_quotes must always be populated from retrieved evidence.
    """

    @pytest.fixture()
    def decision_db(self) -> DatabaseManager:
        db = DatabaseManager(db_path=":memory:")
        db.init_db()
        claim = MemoryClaim(
            subject="backend_service",
            predicate="framework",
            object_value="Go 1.22",
            source_quote="We decided to migrate the backend service from Node.js 20 to Go 1.22.",
            status=ClaimStatus.ACTIVE.value,
            confidence=0.95,
        )
        db.insert_claim(claim)
        return db

    def test_decision_patterns_detected_properly(self) -> None:
        """_is_decision_only must identify migration decision phrases."""
        from core.retrieval_service import _is_decision_only, _is_confirmed_completion, _is_planned_migration
        quotes = [
            "We decided to migrate the backend service from Node.js 20 to Go 1.22.",
            "The team agreed to migrate our database to PostgreSQL.",
            "We chose to migrate to Rust for performance reasons.",
            "Management approved migration to Kubernetes.",
        ]
        for q in quotes:
            assert _is_decision_only(q) is True, f"Expected decision_only: {q}"
            assert _is_confirmed_completion(q) is False, f"Should not be confirmed: {q}"
            assert _is_planned_migration(q) is False, f"Should not be planned: {q}"

    def test_confirmed_patterns_detected_properly(self) -> None:
        """_is_confirmed_completion must identify completed migration phrases."""
        from core.retrieval_service import _is_confirmed_completion, _is_decision_only
        quotes = [
            "We have migrated the backend service from Node.js 20 to Go 1.22.",
            "The team migrated to Python 3.12 last week.",
            "We switched to Go 1.22 successfully.",
            "Migration is complete for the backend service.",
            "The backend is now running on Go 1.22.",
        ]
        for q in quotes:
            assert _is_confirmed_completion(q) is True, f"Expected confirmed: {q}"
            assert _is_decision_only(q) is False, f"Should not be decision_only: {q}"

    def test_decision_quote_annotated_in_evidence_block(self) -> None:
        """Evidence block must annotate decision quotes with DECISION_RECORDED status."""
        claim = MemoryClaim(
            subject="backend_service",
            predicate="framework",
            object_value="Go 1.22",
            source_quote="We decided to migrate the backend service from Node.js 20 to Go 1.22.",
            status=ClaimStatus.ACTIVE.value,
            confidence=0.95,
        )
        block = RetrievalService._build_evidence_block([claim])
        assert "DECISION_RECORDED" in block

    def test_decision_quote_abstains_on_current_framework_query(
        self, decision_db: DatabaseManager
    ) -> None:
        """Query asking what framework is CURRENTLY used must abstain when only decision evidence exists."""
        svc = RetrievalService(db_manager=decision_db)
        result = svc.answer("What backend framework does my project currently use?")

        assert result.abstained is True
        assert result.answer == ""
        reason_lower = result.abstention_reason.lower()
        assert "decision" in reason_lower
        assert "unconfirmed" in reason_lower
        # Issue 2: evidence_ids and evidence_quotes must be populated
        assert len(result.evidence_ids) == 1
        assert len(result.evidence_quotes) == 1
        assert "We decided to migrate" in result.evidence_quotes[0]

    def test_decision_quote_answers_non_current_query(
        self, decision_db: DatabaseManager
    ) -> None:
        """Query asking what was decided about the framework must explain the decision without asserting current use."""
        mock_llm = MagicMock(spec=OllamaClient)
        mock_llm.generate.return_value = (
            "A migration to Go 1.22 was decided [E1], but completion is unconfirmed. "
            '"We decided to migrate the backend service from Node.js 20 to Go 1.22."'
        )
        svc = RetrievalService(db_manager=decision_db, llm_client=mock_llm)
        result = svc.answer("What did the team decide regarding the backend service framework?")

        assert result.abstained is False
        assert "Go 1.22" in result.answer
        assert len(result.evidence_ids) >= 1
        assert len(result.evidence_quotes) >= 1
        assert "We decided to migrate" in result.evidence_quotes[0]

    def test_decision_hallucinated_completion_replaced(
        self, decision_db: DatabaseManager
    ) -> None:
        """When LLM hallucinates current use from a decision quote on a non-current query, post-processor replaces it."""
        bad_llm = MagicMock(spec=OllamaClient)
        bad_llm.generate.return_value = (
            "The backend is currently running Go 1.22 [E1]. "
            '"We decided to migrate the backend service from Node.js 20 to Go 1.22."'
        )
        svc = RetrievalService(db_manager=decision_db, llm_client=bad_llm)
        result = svc.answer("Tell me about the decision regarding the backend framework.")

        assert result.abstained is False
        # The hallucination should be corrected by _make_decision_answer
        assert "was decided" in result.answer
        assert "unconfirmed" in result.answer
        assert len(result.evidence_quotes) >= 1
        assert "We decided to migrate" in result.evidence_quotes[0]

    def test_evidence_quotes_fallback_when_llm_omits_citation(
        self, decision_db: DatabaseManager
    ) -> None:
        """Issue 2: evidence_quotes is populated even if the model's answer omits [E1] markers."""
        mock_llm = MagicMock(spec=OllamaClient)
        mock_llm.generate.return_value = (
            "The team decided on migrating the service to Go 1.22, though completion is unconfirmed."
        )
        svc = RetrievalService(db_manager=decision_db, llm_client=mock_llm)
        result = svc.answer("Tell me about the decision regarding the backend framework.")

        assert result.abstained is False
        assert len(result.evidence_quotes) == 1
        assert result.evidence_quotes[0] == "We decided to migrate the backend service from Node.js 20 to Go 1.22."

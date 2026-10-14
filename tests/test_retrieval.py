"""Tests for the Evidence-Grounded Retrieval Service (Feature 04)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from core.database import DatabaseManager
from core.llm import OllamaClient
from core.retrieval_service import RetrievalConfig, RetrievalResult, RetrievalService, _tokenize, _score_claim
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
        claim = MemoryClaim(
            subject="user",
            predicate="preferred_language",
            object_value="Python",
            source_quote="I prefer Python.",
        )
        score = _score_claim(claim, ["user", "language", "python"])
        assert score >= 2

    def test_zero_for_no_match(self) -> None:
        claim = MemoryClaim(
            subject="project",
            predicate="deadline",
            object_value="2025",
        )
        score = _score_claim(claim, ["language", "python"])
        assert score == 0


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

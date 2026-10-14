"""Tests for RecallGuard memory service."""

from datetime import datetime, timedelta, timezone
import pytest

from core.database import DatabaseManager
from core.memory_service import MemoryService
from core.schemas import MemoryClaim


@pytest.fixture
def service():
    """Fixture providing a MemoryService backed by in-memory SQLite."""
    db = DatabaseManager(":memory:")
    db.init_db()
    return MemoryService(db_manager=db)


def test_service_initialization(service):
    """Verify service initialize creates tables without error."""
    service.initialize()
    stats = service.get_stats()
    assert stats["total"] == 0


def test_service_save_and_get_claim(service):
    """Verify adding and retrieving a claim through the service layer."""
    claim = MemoryClaim(
        subject="user_profile",
        predicate="timezone",
        object_value="UTC+5:30",
        confidence=1.0,
    )
    saved = service.save_claim(claim)
    assert saved.id == claim.id

    fetched = service.get_claim(claim.id)
    assert fetched is not None
    assert fetched.subject == "user_profile"
    assert fetched.predicate == "timezone"
    assert fetched.object_value == "UTC+5:30"


def test_service_list_claims(service):
    """Verify listing claims via service."""
    c1 = MemoryClaim(subject="task", predicate="priority", object_value="high")
    c2 = MemoryClaim(subject="task", predicate="status", object_value="pending")
    service.save_claim(c1)
    service.save_claim(c2)

    claims = service.list_claims()
    assert len(claims) == 2


def test_claim_expiration_logic():
    """Verify temporal expiration calculation on MemoryClaim."""
    past = datetime.now(timezone.utc) - timedelta(days=1)
    future = datetime.now(timezone.utc) + timedelta(days=1)

    expired_claim = MemoryClaim(
        subject="temp_token",
        predicate="val",
        object_value="123",
        valid_until=past,
    )
    active_claim = MemoryClaim(
        subject="temp_token",
        predicate="val",
        object_value="456",
        valid_until=future,
    )
    indefinite_claim = MemoryClaim(
        subject="fact",
        predicate="val",
        object_value="sky_is_blue",
        valid_until=None,
    )

    assert expired_claim.is_expired() is True
    assert active_claim.is_expired() is False
    assert indefinite_claim.is_expired() is False


def test_future_milestones_stubs_raise_not_implemented(service):
    """Verify that advanced AI features are guarded as stubs for Feature 01."""
    claim = MemoryClaim(subject="test", predicate="test", object_value="test")

    with pytest.raises(NotImplementedError) as exc_info1:
        service.extract_claims_from_text("User switched language to Go")
    assert "Feature 02" in str(exc_info1.value)

    with pytest.raises(NotImplementedError) as exc_info2:
        service.resolve_conflicts(claim)
    assert "Feature 03" in str(exc_info2.value)

    with pytest.raises(NotImplementedError) as exc_info3:
        service.retrieve_reliable_memories("current language")
    assert "Feature 04" in str(exc_info3.value)


def test_ollama_client_configuration():
    """Verify local Ollama client configuration defaults to qwen3.5:0.8b."""
    from core.llm import OllamaClient
    client = OllamaClient()
    assert client.model == "qwen3.5:0.8b"
    assert client.host == "http://localhost:11434"
    config = client.config
    assert config.model == "qwen3.5:0.8b"
    assert config.host == "http://localhost:11434"


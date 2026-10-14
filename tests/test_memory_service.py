"""Tests for RecallGuard memory service and local AI extraction."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
import pytest

from core.database import DatabaseManager
from core.llm import OllamaClient
from core.memory_service import MemoryService
from core.schemas import MemoryClaim


@pytest.fixture
def service():
    """Fixture providing a MemoryService backed by in-memory SQLite and a mock LLM."""
    db = DatabaseManager(":memory:")
    db.init_db()
    llm = MagicMock(spec=OllamaClient)
    return MemoryService(db_manager=db, llm_client=llm)


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
        source_quote="My timezone is UTC+5:30",
        confidence=1.0,
    )
    saved = service.save_claim(claim)
    assert saved.id == claim.id

    fetched = service.get_claim(claim.id)
    assert fetched is not None
    assert fetched.subject == "user_profile"
    assert fetched.predicate == "timezone"
    assert fetched.object_value == "UTC+5:30"
    assert fetched.source_quote == "My timezone is UTC+5:30"


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


def test_retrieve_reliable_memories_implemented(service):
    """Feature 04: retrieve_reliable_memories now returns a list (implemented)."""
    # Should return a list (possibly empty) without raising NotImplementedError
    result = service.retrieve_reliable_memories("current language preference")
    assert isinstance(result, list)



def test_ollama_client_configuration():
    """Verify local Ollama client configuration defaults to qwen3.5:0.8b."""
    client = OllamaClient()
    assert client.model == "qwen3.5:0.8b"
    assert client.host == "http://localhost:11434"
    config = client.config
    assert config.model == "qwen3.5:0.8b"
    assert config.host == "http://localhost:11434"


# ==============================================================================
# FEATURE 02 AUTOMATED TESTS: Extraction, Verification, & Persistence
# ==============================================================================


def test_valid_extraction(service):
    """Test successful extraction when model returns well-formed JSON with valid source quotes."""
    conversation = (
        "User: We decided to standardize on PostgreSQL 16 for all new microservices.\n"
        "Assistant: Noted, I will record PostgreSQL 16 as the database standard."
    )
    mock_json = """{
      "claims": [
        {
          "subject": "microservices",
          "predicate": "database_standard",
          "object_value": "PostgreSQL 16",
          "source_quote": "We decided to standardize on PostgreSQL 16 for all new microservices.",
          "confidence": 0.95
        }
      ]
    }"""
    service.llm.chat_structured.return_value = mock_json

    result = service.extract_claims_from_text(conversation, session_id="test-session-1")

    assert result.success is True
    assert len(result.valid_claims) == 1
    assert len(result.rejected_claims) == 0

    claim = result.valid_claims[0]
    assert claim.subject == "microservices"
    assert claim.predicate == "database_standard"
    assert claim.object_value == "PostgreSQL 16"
    assert claim.source_quote == "We decided to standardize on PostgreSQL 16 for all new microservices."
    assert claim.source_session_id == "test-session-1"
    assert claim.confidence == 0.95


def test_malformed_model_output(service):
    """Test safe handling when the model returns unparseable or malformed non-JSON."""
    conversation = "User: I like Python."
    service.llm.chat_structured.return_value = "Sorry, I cannot format this into JSON right now {not valid json"

    result = service.extract_claims_from_text(conversation)

    assert result.success is False
    assert len(result.valid_claims) == 0
    assert len(result.rejected_claims) == 1
    assert "JSON decoding error" in (result.error_message or "")


def test_unsupported_source_quotes(service):
    """Test rejection when model hallucinates a quote not in the input conversation."""
    conversation = "User: Our project will release in November 2026."
    mock_json = """{
      "claims": [
        {
          "subject": "project",
          "predicate": "release_date",
          "object_value": "July 2026",
          "source_quote": "Our project will launch in July 2026.",
          "confidence": 0.9
        }
      ]
    }"""
    service.llm.chat_structured.return_value = mock_json

    result = service.extract_claims_from_text(conversation)

    assert result.success is True
    assert len(result.valid_claims) == 0
    assert len(result.rejected_claims) == 1
    assert "Unsupported source quote" in result.rejected_claims[0].reason
    assert "July 2026" in result.rejected_claims[0].reason


def test_missing_ollama_service(service):
    """Test graceful handling when local Ollama service is unavailable or crashes."""
    conversation = "User: Hello."
    service.llm.chat_structured.side_effect = ConnectionError("Failed to connect to localhost:11434")

    result = service.extract_claims_from_text(conversation)

    assert result.success is False
    assert len(result.valid_claims) == 0
    assert "Local Ollama inference failed" in (result.error_message or "")


def test_database_persistence_with_source_quote(service):
    """Test saving approved claims to SQLite and verifying disk retrieval with source_quote."""
    claim = MemoryClaim(
        subject="api_version",
        predicate="status",
        object_value="deprecated",
        source_quote="The v1 API is deprecated as of today.",
        confidence=0.99,
    )
    saved = service.save_claim(claim)
    assert saved.id == claim.id

    fetched = service.get_claim(claim.id)
    assert fetched is not None
    assert fetched.source_quote == "The v1 API is deprecated as of today."
    assert fetched.confidence == 0.99


def test_save_batch_claims(service):
    """Test saving multiple approved claims in batch."""
    claims = [
        MemoryClaim(subject="s1", predicate="p1", object_value="o1", source_quote="q1"),
        MemoryClaim(subject="s2", predicate="p2", object_value="o2", source_quote="q2"),
    ]
    saved = service.save_claims(claims)
    assert len(saved) == 2
    assert service.get_stats()["total"] == 2


def test_verify_quote_in_text_edge_cases():
    """Test quote verification with case variations and collapsed whitespaces."""
    text = "User said:   We MUST use   Docker containers   everywhere!"
    assert MemoryService.verify_quote_in_text("we must use docker containers everywhere!", text) is True
    assert MemoryService.verify_quote_in_text("We MUST use Docker containers", text) is True
    assert MemoryService.verify_quote_in_text("We use Kubernetes everywhere", text) is False
    assert MemoryService.verify_quote_in_text("", text) is False

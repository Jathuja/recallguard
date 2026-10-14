"""Tests for RecallGuard database management."""

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import pytest

from core.database import DatabaseManager
from core.schemas import MemoryClaim


@pytest.fixture
def memory_db():
    """Fixture providing an initialized in-memory database."""
    db = DatabaseManager(":memory:")
    db.init_db()
    return db


@pytest.fixture
def temp_file_db():
    """Fixture providing a temporary file-based database."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_recallguard.db"
        db = DatabaseManager(db_path)
        db.init_db()
        yield db


def test_database_initialization_idempotent(memory_db):
    """Verify that calling init_db repeatedly is safe and idempotent."""
    # Calling init_db multiple times should not raise errors
    memory_db.init_db()
    memory_db.init_db()

    with memory_db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        assert "memory_claims" in tables
        assert "schema_version" in tables


def test_insert_and_retrieve_claim(memory_db):
    """Verify storing and retrieving a memory claim."""
    claim = MemoryClaim(
        subject="agent_runtime",
        predicate="primary_language",
        object_value="Python 3.12",
        confidence=0.98,
        source_session_id="session-42",
        metadata={"tags": ["core", "config"]},
    )

    saved = memory_db.insert_claim(claim)
    assert saved.id == claim.id

    retrieved = memory_db.get_claim(claim.id)
    assert retrieved is not None
    assert retrieved.id == claim.id
    assert retrieved.subject == "agent_runtime"
    assert retrieved.predicate == "primary_language"
    assert retrieved.object_value == "Python 3.12"
    assert retrieved.confidence == 0.98
    assert retrieved.source_session_id == "session-42"
    assert retrieved.is_active is True
    assert retrieved.metadata == {"tags": ["core", "config"]}


def test_list_claims_filtering(memory_db):
    """Verify listing claims with active filtering."""
    c1 = MemoryClaim(
        subject="model",
        predicate="version",
        object_value="0.8b",
        is_active=True,
    )
    c2 = MemoryClaim(
        subject="model",
        predicate="version",
        object_value="0.5b",
        is_active=False,
    )
    memory_db.insert_claim(c1)
    memory_db.insert_claim(c2)

    all_claims = memory_db.list_claims(active_only=False)
    assert len(all_claims) == 2

    active_claims = memory_db.list_claims(active_only=True)
    assert len(active_claims) == 1
    assert active_claims[0].id == c1.id


def test_count_claims(memory_db):
    """Verify claim counts calculation."""
    counts = memory_db.count_claims()
    assert counts == {"total": 0, "active": 0, "inactive": 0}

    memory_db.insert_claim(
        MemoryClaim(subject="sys", predicate="status", object_value="ok", is_active=True)
    )
    memory_db.insert_claim(
        MemoryClaim(subject="sys", predicate="deprecated", object_value="yes", is_active=False)
    )

    counts = memory_db.count_claims()
    assert counts["total"] == 2
    assert counts["active"] == 1
    assert counts["inactive"] == 1


def test_reset_db(memory_db):
    """Verify that reset_db drops and recreates empty tables."""
    memory_db.insert_claim(
        MemoryClaim(subject="test", predicate="state", object_value="temporary")
    )
    assert memory_db.count_claims()["total"] == 1

    memory_db.reset_db()
    assert memory_db.count_claims()["total"] == 0


def test_file_based_db_persistence(temp_file_db):
    """Verify that file-based database creates parent directories and persists records."""
    assert Path(temp_file_db.db_path).exists()
    claim = MemoryClaim(
        subject="persistent",
        predicate="location",
        object_value="disk",
    )
    temp_file_db.insert_claim(claim)

    # Re-instantiate from the same file path to verify disk persistence
    db2 = DatabaseManager(temp_file_db.db_path)
    loaded = db2.get_claim(claim.id)
    assert loaded is not None
    assert loaded.object_value == "disk"

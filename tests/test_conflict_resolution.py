"""Tests for Feature 03: Temporal Memory Conflict Resolution."""

from datetime import datetime, timezone
import pytest

from core.conflict_detector import (
    detect_transition_in_quote,
    evaluate_claim_conflict,
)
from core.database import DatabaseManager
from core.memory_service import MemoryService
from core.schemas import ClaimStatus, ConflictType, MemoryClaim


@pytest.fixture
def service():
    """Fixture providing a fresh in-memory database and MemoryService."""
    db = DatabaseManager(":memory:")
    db.init_db()
    return MemoryService(db_manager=db)


def test_no_conflict_when_store_empty(service):
    """Verify that a claim with no prior records resolves as NO_CONFLICT."""
    new_claim = MemoryClaim(
        subject="backend_service",
        predicate="framework",
        object_value="FastAPI",
        source_quote="We built the service with FastAPI.",
    )
    decision = service.resolve_conflicts(new_claim)
    assert decision.conflict_type == ConflictType.NO_CONFLICT.value
    assert decision.recommended_action == "add_new"
    assert decision.existing_claim is None


def test_duplicate_claims_detection(service):
    """Verify that identical subject, predicate, and object_value resolve as EXACT_DUPLICATE."""
    existing = MemoryClaim(
        subject="backend_service",
        predicate="primary_database",
        object_value="PostgreSQL 16",
        source_quote="We use PostgreSQL 16.",
        status=ClaimStatus.ACTIVE.value,
    )
    service.save_claim(existing)

    duplicate_candidate = MemoryClaim(
        subject="backend_service",
        predicate="primary_database",
        object_value="postgresql 16",  # Case insensitive duplicate
        source_quote="Our database is postgresql 16.",
    )
    decision = service.resolve_conflicts(duplicate_candidate)
    assert decision.conflict_type == ConflictType.EXACT_DUPLICATE.value
    assert decision.recommended_action == "keep_existing"
    assert decision.existing_claim is not None
    assert decision.existing_claim.id == existing.id


def test_explicit_update_with_verified_evidence(service):
    """Verify that transition language explicitly referencing old and new values triggers EXPLICIT_UPDATE."""
    old_claim = MemoryClaim(
        subject="backend",
        predicate="framework",
        object_value="Node.js",
        source_quote="Our backend is running on Node.js.",
        status=ClaimStatus.ACTIVE.value,
    )
    service.save_claim(old_claim)

    # Transition sentence explicitly indicating migration
    update_claim = MemoryClaim(
        subject="backend",
        predicate="framework",
        object_value="Go 1.22",
        source_quote="We migrated the backend from Node.js to Go 1.22.",
    )

    decision = service.resolve_conflicts(update_claim)
    assert decision.conflict_type == ConflictType.EXPLICIT_UPDATE.value
    assert decision.is_explicit_update is True
    assert decision.recommended_action == "supersede_existing"
    assert "Explicit update detected" in decision.decision_reason
    assert decision.existing_claim.id == old_claim.id


def test_ambiguous_update_mere_mention(service):
    """Verify that a mere mention of a different value without transition language is marked AMBIGUOUS_CHANGE."""
    old_claim = MemoryClaim(
        subject="backend",
        predicate="framework",
        object_value="Node.js",
        source_quote="Our backend is running on Node.js.",
        status=ClaimStatus.ACTIVE.value,
    )
    service.save_claim(old_claim)

    # Mere mention of Go 1.22 without words like migrated, switched, updated, replaced, etc.
    mere_mention_claim = MemoryClaim(
        subject="backend",
        predicate="framework",
        object_value="Go 1.22",
        source_quote="Go 1.22 has great garbage collection performance.",
    )

    decision = service.resolve_conflicts(mere_mention_claim)
    assert decision.conflict_type == ConflictType.AMBIGUOUS_CHANGE.value
    assert decision.is_explicit_update is False
    assert decision.recommended_action == "needs_review"
    assert "Ambiguous conflict" in decision.decision_reason


def test_missing_or_invalid_evidence(service):
    """Verify that claims with missing or empty source quotes are flagged as INVALID_EVIDENCE."""
    old_claim = MemoryClaim(
        subject="user_profile",
        predicate="theme",
        object_value="dark",
        source_quote="I use dark mode.",
        status=ClaimStatus.ACTIVE.value,
    )
    service.save_claim(old_claim)

    invalid_claim = MemoryClaim(
        subject="user_profile",
        predicate="theme",
        object_value="light",
        source_quote="",  # Empty evidence
    )

    decision = service.resolve_conflicts(invalid_claim)
    assert decision.conflict_type == ConflictType.INVALID_EVIDENCE.value
    assert decision.recommended_action == "needs_review"
    assert "missing or empty" in decision.decision_reason


def test_approving_a_change_supersedes_atomically(service):
    """Verify that approving a proposed update atomically marks the old record superseded and activates the new one."""
    old_claim = MemoryClaim(
        subject="api_auth",
        predicate="auth_method",
        object_value="API_KEY",
        source_quote="Use API_KEY for authenticating.",
        status=ClaimStatus.ACTIVE.value,
    )
    service.save_claim(old_claim)

    new_claim = MemoryClaim(
        subject="api_auth",
        predicate="auth_method",
        object_value="OAuth2_Bearer",
        source_quote="We switched from API_KEY to OAuth2_Bearer.",
        status=ClaimStatus.NEEDS_REVIEW.value,
    )

    # Approve update
    approved = service.approve_conflict_update(
        proposed_claim=new_claim,
        existing_claim_id=old_claim.id,
    )

    # Check new claim is active
    assert approved.status == ClaimStatus.ACTIVE.value
    assert approved.is_active is True
    assert approved.supersedes_id == old_claim.id

    # Check old claim is superseded and NOT deleted
    old_retrieved = service.get_claim(old_claim.id)
    assert old_retrieved is not None
    assert old_retrieved.status == ClaimStatus.SUPERSEDED.value
    assert old_retrieved.is_active is False
    assert old_retrieved.superseded_by == new_claim.id
    assert old_retrieved.valid_until is not None


def test_rejecting_a_change_preserves_existing_claim(service):
    """Verify that rejecting a proposed change keeps the existing claim active and archives the proposed claim."""
    old_claim = MemoryClaim(
        subject="project",
        predicate="release_quarter",
        object_value="Q3 2026",
        source_quote="Targeting release in Q3 2026.",
        status=ClaimStatus.ACTIVE.value,
    )
    service.save_claim(old_claim)

    proposed_claim = MemoryClaim(
        subject="project",
        predicate="release_quarter",
        object_value="Q1 2026",
        source_quote="Maybe we can ship in Q1 2026.",
        status=ClaimStatus.NEEDS_REVIEW.value,
    )
    service.save_claim(proposed_claim)

    # Reject proposed update
    success = service.reject_conflict_update(
        proposed_claim_id=proposed_claim.id,
        reason="Leadership rejected expedited schedule",
    )
    assert success is True

    # Existing claim must remain active
    old_retrieved = service.get_claim(old_claim.id)
    assert old_retrieved.status == ClaimStatus.ACTIVE.value
    assert old_retrieved.is_active is True

    # Proposed claim must be superseded / inactive
    prop_retrieved = service.get_claim(proposed_claim.id)
    assert prop_retrieved.status == ClaimStatus.SUPERSEDED.value
    assert prop_retrieved.is_active is False
    assert prop_retrieved.metadata["status_reason"] == "Leadership rejected expedited schedule"


def test_preserving_historical_records_complete_audit(service):
    """Verify that historical records and their quotes are never deleted after multiple updates."""
    # Step 1: Initial claim
    c1 = MemoryClaim(
        subject="infrastructure",
        predicate="cloud_provider",
        object_value="AWS",
        source_quote="We host on AWS.",
    )
    service.save_claim(c1)

    # Step 2: First migration to GCP
    c2 = MemoryClaim(
        subject="infrastructure",
        predicate="cloud_provider",
        object_value="GCP",
        source_quote="We migrated our infrastructure from AWS to GCP.",
    )
    service.approve_conflict_update(c2, c1.id)

    # Step 3: Second migration to Bare Metal
    c3 = MemoryClaim(
        subject="infrastructure",
        predicate="cloud_provider",
        object_value="BareMetal",
        source_quote="We moved all workloads from GCP to our BareMetal data center.",
    )
    service.approve_conflict_update(c3, c2.id)

    # Total claims must be 3; only c3 is active
    stats = service.get_stats()
    assert stats["total"] == 3
    assert stats["active"] == 1
    assert stats["superseded"] == 2

    # Verify all source quotes remain intact
    r1 = service.get_claim(c1.id)
    r2 = service.get_claim(c2.id)
    r3 = service.get_claim(c3.id)

    assert r1.source_quote == "We host on AWS."
    assert r1.superseded_by == c2.id

    assert r2.source_quote == "We migrated our infrastructure from AWS to GCP."
    assert r2.supersedes_id == c1.id
    assert r2.superseded_by == c3.id

    assert r3.source_quote == "We moved all workloads from GCP to our BareMetal data center."
    assert r3.supersedes_id == c2.id
    assert r3.status == ClaimStatus.ACTIVE.value


def test_staging_and_review_queue(service):
    """Verify process_and_stage_claim places ambiguous changes into the review queue."""
    existing = MemoryClaim(
        subject="ide",
        predicate="indent_size",
        object_value="2",
        source_quote="Indent size is 2 spaces.",
        status=ClaimStatus.ACTIVE.value,
    )
    service.save_claim(existing)

    ambiguous = MemoryClaim(
        subject="ide",
        predicate="indent_size",
        object_value="4",
        source_quote="I set tab width to 4.",
    )
    staged, decision = service.process_and_stage_claim(ambiguous, auto_supersede_explicit=False)
    assert staged.status == ClaimStatus.NEEDS_REVIEW.value
    assert staged.is_active is False

    # Review queue should list this item
    queue = service.list_review_queue()
    assert len(queue) == 1
    assert queue[0]["pending_claim"].id == staged.id
    assert queue[0]["existing_claim"].id == existing.id


def test_sqlite_migration_with_status_columns():
    """Verify non-destructive migration of legacy SQLite schema lacking status and supersession columns."""
    import sqlite3
    db = DatabaseManager(":memory:")
    # Create legacy table with only Feature 01/02 columns
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE memory_claims (
                id TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object_value TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                source_quote TEXT,
                source_session_id TEXT,
                valid_from TEXT,
                valid_until TEXT,
                created_at TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                metadata TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        cursor.execute(
            """
            INSERT INTO memory_claims (
                id, subject, predicate, object_value, created_at, is_active
            ) VALUES ('legacy-1', 'sys', 'state', 'active', '2026-01-01', 1)
            """
        )
        cursor.execute(
            """
            INSERT INTO memory_claims (
                id, subject, predicate, object_value, created_at, is_active
            ) VALUES ('legacy-2', 'sys', 'state', 'obsolete', '2026-01-01', 0)
            """
        )

    # Running init_db must safely add status, superseded_by, supersedes_id without error
    db.init_db()

    # Verify columns were added
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(memory_claims)")
        col_names = {r["name"] for r in cursor.fetchall()}
        assert "status" in col_names
        assert "superseded_by" in col_names
        assert "supersedes_id" in col_names

    # Verify backfilled status values
    c1 = db.get_claim("legacy-1")
    c2 = db.get_claim("legacy-2")
    assert c1.status == ClaimStatus.ACTIVE.value
    assert c2.status == ClaimStatus.SUPERSEDED.value

"""Safe and repeatable SQLite database initialization and access for RecallGuard."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, Generator, List, Optional, Union
from contextlib import contextmanager

from core.schemas import ClaimStatus, MemoryClaim

DEFAULT_DB_PATH = Path("data/recallguard.db")

TABLE_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_claims (
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
    status TEXT NOT NULL DEFAULT 'active',
    is_active INTEGER NOT NULL DEFAULT 1,
    superseded_by TEXT,
    supersedes_id TEXT,
    metadata TEXT NOT NULL DEFAULT '{}'
);
"""

INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_claims_subject_predicate 
ON memory_claims(subject, predicate);

CREATE INDEX IF NOT EXISTS idx_claims_status 
ON memory_claims(status);

CREATE INDEX IF NOT EXISTS idx_claims_is_active 
ON memory_claims(is_active);

CREATE INDEX IF NOT EXISTS idx_claims_valid_until 
ON memory_claims(valid_until);
"""


class DatabaseManager:
    """Manages SQLite connections, schema initialization, and transactional claim lifecycle."""

    def __init__(self, db_path: Union[str, Path] = DEFAULT_DB_PATH):
        self.db_path = str(db_path)
        self._is_memory = self.db_path == ":memory:"
        self._memory_conn: Optional[sqlite3.Connection] = None
        if not self._is_memory:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Provide a transactional SQLite connection."""
        if self._is_memory:
            if self._memory_conn is None:
                self._memory_conn = sqlite3.connect(":memory:")
                self._memory_conn.row_factory = sqlite3.Row
            yield self._memory_conn
            return

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db(self) -> None:
        """Safely and repeatably initialize database schema with non-destructive migrations."""
        if not self._is_memory:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            # 1. Create base tables
            cursor.executescript(TABLE_SQL)

            # 2. Safe non-destructive column migrations on existing databases
            cursor.execute("PRAGMA table_info(memory_claims)")
            columns = {row["name"] for row in cursor.fetchall()}

            if "source_quote" not in columns:
                cursor.execute("ALTER TABLE memory_claims ADD COLUMN source_quote TEXT")
            if "status" not in columns:
                cursor.execute("ALTER TABLE memory_claims ADD COLUMN status TEXT NOT NULL DEFAULT 'active'")
                # Sync existing is_active flags to status
                cursor.execute("UPDATE memory_claims SET status = 'superseded' WHERE is_active = 0")
                cursor.execute("UPDATE memory_claims SET status = 'active' WHERE is_active = 1")
            if "superseded_by" not in columns:
                cursor.execute("ALTER TABLE memory_claims ADD COLUMN superseded_by TEXT")
            if "supersedes_id" not in columns:
                cursor.execute("ALTER TABLE memory_claims ADD COLUMN supersedes_id TEXT")

            # 3. Create indexes now that all columns are guaranteed to exist
            cursor.executescript(INDEX_SQL)

            # Record initial schema version if not present
            cursor.execute("SELECT COUNT(*) FROM schema_version WHERE version = 1")
            if cursor.fetchone()[0] == 0:
                cursor.execute(
                    "INSERT INTO schema_version (version, applied_at) VALUES (1, datetime('now'))"
                )
            if self._is_memory:
                conn.commit()

    def reset_db(self) -> None:
        """Drop existing tables and re-initialize schema (primarily for testing)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DROP TABLE IF EXISTS memory_claims")
            cursor.execute("DROP TABLE IF EXISTS schema_version")
            if self._is_memory:
                conn.commit()
        self.init_db()

    def insert_claim(self, claim: MemoryClaim) -> MemoryClaim:
        """Insert or replace a memory claim in the database."""
        # Ensure is_active flag matches status
        is_active_int = 1 if claim.status == ClaimStatus.ACTIVE.value and claim.is_active else 0

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO memory_claims (
                    id, subject, predicate, object_value, confidence,
                    source_quote, source_session_id, valid_from, valid_until, created_at,
                    status, is_active, superseded_by, supersedes_id, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    claim.id,
                    claim.subject,
                    claim.predicate,
                    claim.object_value,
                    claim.confidence,
                    claim.source_quote,
                    claim.source_session_id,
                    claim.valid_from.isoformat() if claim.valid_from else None,
                    claim.valid_until.isoformat() if claim.valid_until else None,
                    claim.created_at.isoformat(),
                    claim.status,
                    is_active_int,
                    claim.superseded_by,
                    claim.supersedes_id,
                    json.dumps(claim.metadata),
                ),
            )
            if self._is_memory:
                conn.commit()
        return claim

    def get_claim(self, claim_id: str) -> Optional[MemoryClaim]:
        """Retrieve a single claim by its ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM memory_claims WHERE id = ?",
                (claim_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_claim(row)

    def list_claims(
        self, active_only: bool = False, status: Optional[str] = None
    ) -> List[MemoryClaim]:
        """List claims with optional active or status filtering."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if status:
                cursor.execute(
                    "SELECT * FROM memory_claims WHERE status = ? ORDER BY created_at DESC",
                    (status,),
                )
            elif active_only:
                cursor.execute(
                    "SELECT * FROM memory_claims WHERE status = 'active' AND is_active = 1 ORDER BY created_at DESC"
                )
            else:
                cursor.execute("SELECT * FROM memory_claims ORDER BY created_at DESC")
            rows = cursor.fetchall()
            return [self._row_to_claim(row) for row in rows]

    def find_matching_claims(
        self, subject: str, predicate: str, active_only: bool = True
    ) -> List[MemoryClaim]:
        """Find claims with matching normalized subject and predicate."""
        norm_subj = subject.strip().lower()
        norm_pred = predicate.strip().lower()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if active_only:
                cursor.execute(
                    """
                    SELECT * FROM memory_claims 
                    WHERE LOWER(TRIM(subject)) = ? AND LOWER(TRIM(predicate)) = ? AND status = 'active' AND is_active = 1
                    ORDER BY created_at DESC
                    """,
                    (norm_subj, norm_pred),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM memory_claims 
                    WHERE LOWER(TRIM(subject)) = ? AND LOWER(TRIM(predicate)) = ?
                    ORDER BY created_at DESC
                    """,
                    (norm_subj, norm_pred),
                )
            rows = cursor.fetchall()
            return [self._row_to_claim(row) for row in rows]

    def supersede_claim_atomic(
        self,
        old_claim_id: str,
        new_claim: MemoryClaim,
        superseded_at: Optional[datetime] = None,
    ) -> MemoryClaim:
        """Atomically supersede an older claim and activate a replacement claim in one transaction.
        
        Preserves the older claim, marks it as 'superseded' with valid_until and superseded_by,
        and inserts the replacement claim as 'active' with valid_from and supersedes_id.
        """
        ts = superseded_at or datetime.now(timezone.utc)
        ts_iso = ts.isoformat()

        new_claim.status = ClaimStatus.ACTIVE.value
        new_claim.is_active = True
        new_claim.valid_from = ts
        new_claim.supersedes_id = old_claim_id

        with self.get_connection() as conn:
            cursor = conn.cursor()
            # 1. Update old claim to superseded
            cursor.execute(
                """
                UPDATE memory_claims 
                SET status = 'superseded', is_active = 0, valid_until = ?, superseded_by = ?
                WHERE id = ?
                """,
                (ts_iso, new_claim.id, old_claim_id),
            )

            # 2. Insert new replacement claim
            cursor.execute(
                """
                INSERT OR REPLACE INTO memory_claims (
                    id, subject, predicate, object_value, confidence,
                    source_quote, source_session_id, valid_from, valid_until, created_at,
                    status, is_active, superseded_by, supersedes_id, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_claim.id,
                    new_claim.subject,
                    new_claim.predicate,
                    new_claim.object_value,
                    new_claim.confidence,
                    new_claim.source_quote,
                    new_claim.source_session_id,
                    new_claim.valid_from.isoformat() if new_claim.valid_from else None,
                    new_claim.valid_until.isoformat() if new_claim.valid_until else None,
                    new_claim.created_at.isoformat(),
                    new_claim.status,
                    1,
                    new_claim.superseded_by,
                    new_claim.supersedes_id,
                    json.dumps(new_claim.metadata),
                ),
            )
            if self._is_memory:
                conn.commit()

        return new_claim

    def update_claim_status(
        self,
        claim_id: str,
        status: str,
        is_active: Optional[bool] = None,
        reason: Optional[str] = None,
    ) -> bool:
        """Update claim status and metadata reason safely."""
        active_val = (status == ClaimStatus.ACTIVE.value) if is_active is None else is_active
        is_active_int = 1 if active_val else 0

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT metadata FROM memory_claims WHERE id = ?", (claim_id,))
            row = cursor.fetchone()
            if not row:
                return False

            meta = json.loads(row["metadata"] or "{}")
            if reason:
                meta["status_reason"] = reason
                meta["status_updated_at"] = datetime.now(timezone.utc).isoformat()

            cursor.execute(
                """
                UPDATE memory_claims 
                SET status = ?, is_active = ?, metadata = ?
                WHERE id = ?
                """,
                (status, is_active_int, json.dumps(meta), claim_id),
            )
            if self._is_memory:
                conn.commit()
            return True

    def count_claims(self, extended: bool = False) -> Dict[str, int]:
        """Return counts of claims, preserving total/active/inactive baseline."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM memory_claims")
            total = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM memory_claims WHERE status = 'active' AND is_active = 1")
            active = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM memory_claims WHERE status = 'superseded'")
            superseded = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM memory_claims WHERE status = 'needs_review'")
            needs_review = cursor.fetchone()[0]

            res = {
                "total": total,
                "active": active,
                "inactive": total - active,
            }
            if extended:
                res["superseded"] = superseded
                res["needs_review"] = needs_review
            return res

    def count_claims_detailed(self) -> Dict[str, int]:
        """Return counts including status breakdown."""
        return self.count_claims(extended=True)

    @staticmethod
    def _row_to_claim(row: sqlite3.Row) -> MemoryClaim:
        """Convert a SQLite row to a MemoryClaim model with backward compatibility."""
        row_keys = row.keys()
        metadata_dict = json.loads(row["metadata"] or "{}")

        source_quote = None
        if "source_quote" in row_keys:
            source_quote = row["source_quote"]
        if not source_quote and "source_quote" in metadata_dict:
            source_quote = metadata_dict["source_quote"]

        # Handle status column migration backward compatibility
        status = ClaimStatus.ACTIVE.value
        if "status" in row_keys and row["status"]:
            status = row["status"]
        elif "is_active" in row_keys and not row["is_active"]:
            status = ClaimStatus.SUPERSEDED.value

        superseded_by = row["superseded_by"] if "superseded_by" in row_keys else None
        supersedes_id = row["supersedes_id"] if "supersedes_id" in row_keys else None

        return MemoryClaim(
            id=row["id"],
            subject=row["subject"],
            predicate=row["predicate"],
            object_value=row["object_value"],
            confidence=row["confidence"],
            source_quote=source_quote,
            source_session_id=row["source_session_id"],
            valid_from=row["valid_from"],
            valid_until=row["valid_until"],
            created_at=row["created_at"],
            status=status,
            is_active=bool(row["is_active"]),
            superseded_by=superseded_by,
            supersedes_id=supersedes_id,
            metadata=metadata_dict,
        )

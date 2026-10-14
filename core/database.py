"""Safe and repeatable SQLite database initialization and access for RecallGuard."""

import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, Generator, List, Optional, Union
from contextlib import contextmanager

from core.schemas import MemoryClaim

DEFAULT_DB_PATH = Path("data/recallguard.db")

SCHEMA_SQL = """
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
    source_session_id TEXT,
    valid_from TEXT,
    valid_until TEXT,
    created_at TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1,
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_claims_subject_predicate 
ON memory_claims(subject, predicate);

CREATE INDEX IF NOT EXISTS idx_claims_is_active 
ON memory_claims(is_active);

CREATE INDEX IF NOT EXISTS idx_claims_valid_until 
ON memory_claims(valid_until);
"""


class DatabaseManager:
    """Manages SQLite connections, schema initialization, and basic claim persistence."""

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
        """Safely and repeatably initialize the database schema."""
        if not self._is_memory:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executescript(SCHEMA_SQL)
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
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO memory_claims (
                    id, subject, predicate, object_value, confidence,
                    source_session_id, valid_from, valid_until, created_at,
                    is_active, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    claim.id,
                    claim.subject,
                    claim.predicate,
                    claim.object_value,
                    claim.confidence,
                    claim.source_session_id,
                    claim.valid_from.isoformat() if claim.valid_from else None,
                    claim.valid_until.isoformat() if claim.valid_until else None,
                    claim.created_at.isoformat(),
                    1 if claim.is_active else 0,
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

    def list_claims(self, active_only: bool = False) -> List[MemoryClaim]:
        """List all claims, optionally filtering by active state."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if active_only:
                cursor.execute(
                    "SELECT * FROM memory_claims WHERE is_active = 1 ORDER BY created_at DESC"
                )
            else:
                cursor.execute("SELECT * FROM memory_claims ORDER BY created_at DESC")
            rows = cursor.fetchall()
            return [self._row_to_claim(row) for row in rows]

    def count_claims(self) -> Dict[str, int]:
        """Return counts of total, active, and inactive claims."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM memory_claims")
            total = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM memory_claims WHERE is_active = 1")
            active = cursor.fetchone()[0]
            return {
                "total": total,
                "active": active,
                "inactive": total - active,
            }

    @staticmethod
    def _row_to_claim(row: sqlite3.Row) -> MemoryClaim:
        """Convert a SQLite row to a MemoryClaim model."""
        return MemoryClaim(
            id=row["id"],
            subject=row["subject"],
            predicate=row["predicate"],
            object_value=row["object_value"],
            confidence=row["confidence"],
            source_session_id=row["source_session_id"],
            valid_from=row["valid_from"],
            valid_until=row["valid_until"],
            created_at=row["created_at"],
            is_active=bool(row["is_active"]),
            metadata=json.loads(row["metadata"] or "{}"),
        )

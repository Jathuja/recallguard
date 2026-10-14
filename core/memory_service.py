"""Memory Service for managing claims, storage, and foundational lifecycle operations."""

from typing import List, Optional
from core.database import DatabaseManager, DEFAULT_DB_PATH
from core.llm import OllamaClient
from core.schemas import MemoryClaim


class MemoryService:
    """Foundational service coordinating memory claims persistence and model interaction.
    
    Milestone 1 implements claim persistence and health checks.
    Advanced extraction, temporal conflict resolution, and reliability evaluation
    are reserved for subsequent milestones.
    """

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        llm_client: Optional[OllamaClient] = None,
    ):
        self.db = db_manager or DatabaseManager(DEFAULT_DB_PATH)
        self.llm = llm_client or OllamaClient()

    def initialize(self) -> None:
        """Initialize database schema safely."""
        self.db.init_db()

    def save_claim(self, claim: MemoryClaim) -> MemoryClaim:
        """Store or update a memory claim in the database."""
        return self.db.insert_claim(claim)

    def get_claim(self, claim_id: str) -> Optional[MemoryClaim]:
        """Fetch a single claim by identifier."""
        return self.db.get_claim(claim_id)

    def list_claims(self, active_only: bool = False) -> List[MemoryClaim]:
        """Retrieve stored memory claims."""
        return self.db.list_claims(active_only=active_only)

    def get_stats(self) -> dict:
        """Return memory store statistics."""
        return self.db.count_claims()

    # --- Foundation Stubs for Upcoming Milestones ---

    def extract_claims_from_text(self, text: str, session_id: Optional[str] = None) -> List[MemoryClaim]:
        """Stub: AI claim extraction will be implemented in Feature 02."""
        raise NotImplementedError(
            "AI extraction logic is scheduled for implementation in Feature 02."
        )

    def resolve_conflicts(self, new_claim: MemoryClaim) -> List[MemoryClaim]:
        """Stub: Temporal conflict resolution will be implemented in Feature 03."""
        raise NotImplementedError(
            "Conflict resolution logic is scheduled for implementation in Feature 03."
        )

    def retrieve_reliable_memories(self, query: str) -> List[MemoryClaim]:
        """Stub: Temporal reliability retrieval will be implemented in Feature 04."""
        raise NotImplementedError(
            "Reliability retrieval logic is scheduled for implementation in Feature 04."
        )

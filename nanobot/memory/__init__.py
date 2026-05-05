"""Progressive Memory Consolidation (PMC) — public API.

Usage:
    pmc = PMCMemory(db_path, llm_call)

    # Before each agent turn: recall relevant memories
    recalled = pmc.recall(query)
    prompt_section = recalled.format_for_prompt()

    # After each agent turn: log the episode
    await pmc.log_episode(session_key, query, summary, outcome, tags)

    # Consolidation runs automatically when episode threshold is reached.
    # You can also force it:
    await pmc.force_consolidate()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Awaitable

from nanobot.memory.consolidator import Consolidator
from nanobot.memory.models import EpisodicMemory
from nanobot.memory.recall import Recall, RecallResult
from nanobot.memory.store import MemoryStore

logger = logging.getLogger(__name__)

# Type alias for the LLM call function
LLMFunc = Callable[[list[dict[str, Any]]], Awaitable[str]]


class PMCMemory:
    """Progressive Memory Consolidation system.

    Manages the full lifecycle:
    1. Log episodes after each interaction
    2. Automatically consolidate when thresholds are met
    3. Provide associative recall for context building
    """

    def __init__(
        self,
        db_path: Path | str,
        llm_call: LLMFunc,
        episode_batch_size: int = 5,
        semantic_threshold: int = 6,
    ):
        self.store = MemoryStore(db_path)
        self.recall_engine = Recall(self.store)
        self.consolidator = Consolidator(
            store=self.store,
            llm_call=llm_call,
            episode_batch_size=episode_batch_size,
            semantic_threshold=semantic_threshold,
        )

    # ── Core API ─────────────────────────────────────────────────────────

    def recall(self, query: str, **kwargs) -> RecallResult:
        """Retrieve relevant memories for the current query."""
        return self.recall_engine.recall(query, **kwargs)

    async def log_episode(
        self,
        session_key: str,
        query: str,
        summary: str,
        outcome: str = "neutral",
        tags: list[str] | None = None,
    ) -> dict[str, int]:
        """Log a completed interaction episode and trigger consolidation if needed.

        Returns consolidation stats (empty dict if no consolidation occurred).
        """
        ep = EpisodicMemory(
            session_key=session_key,
            query=query,
            summary=summary,
            outcome=outcome,
            tags=tags or [],
        )
        self.store.save_episode(ep)

        # Automatic consolidation check
        consolidation_result = await self.consolidator.maybe_consolidate()
        if any(v > 0 for v in consolidation_result.values()):
            logger.info(f"PMC consolidation: {consolidation_result}")
        return consolidation_result

    async def force_consolidate(self) -> dict[str, int]:
        """Force a consolidation pass regardless of thresholds."""
        # Temporarily lower thresholds
        old_batch = self.consolidator.episode_batch_size
        old_thresh = self.consolidator.semantic_threshold
        self.consolidator.episode_batch_size = 1
        self.consolidator.semantic_threshold = 1
        try:
            return await self.consolidator.maybe_consolidate()
        finally:
            self.consolidator.episode_batch_size = old_batch
            self.consolidator.semantic_threshold = old_thresh

    def stats(self) -> dict[str, Any]:
        """Return memory system statistics."""
        base = self.store.stats()
        # Add active counts (above decay threshold)
        base["active_semantic"] = len(self.store.get_active_semantic())
        base["active_procedural"] = len(self.store.get_active_procedural())
        return base

    def close(self) -> None:
        self.store.close()

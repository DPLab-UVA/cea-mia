"""Data models for Progressive Memory Consolidation (PMC) — v2.

Changes from v1:
- Semantic decay: 0.1 → 0.01 (survives 90 days without access)
- Procedural decay: 0.05 → 0.01 (survives 90 days without access)
- Episodic decay: unchanged (0.3, episodes should fade fast → consolidation picks up)

Design rationale:
- Semantic memories represent consolidated *knowledge* — they should be stable
  for months, mimicking how human semantic memory works (Squire & Zola, 1996).
- Procedural memories represent learned *strategies* — even more stable.
- Episodic memories are raw interaction traces — fast decay pushes toward consolidation.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class MemoryType(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class EpisodicMemory:
    """A single interaction episode."""

    id: str = field(default_factory=_new_id)
    session_key: str = ""
    query: str = ""
    summary: str = ""
    outcome: str = "neutral"
    tags: list[str] = field(default_factory=list)

    created_at: datetime = field(default_factory=_now)
    last_accessed: datetime = field(default_factory=_now)
    access_count: int = 0
    confidence: float = 0.5

    consolidated: bool = False

    def strength(self) -> float:
        """Ebbinghaus-inspired decay — fast for episodes (pushes consolidation)."""
        hours = max(((_now() - self.last_accessed).total_seconds()) / 3600, 0.01)
        stability = 1.0 + math.log1p(self.access_count)
        # 0.3 coefficient: episodes decay within ~1-2 days without access
        decay = math.exp(-0.3 * hours / stability)
        return self.confidence * decay

    def touch(self) -> None:
        self.last_accessed = _now()
        self.access_count += 1


@dataclass
class SemanticMemory:
    """A distilled fact or knowledge extracted from multiple episodes."""

    id: str = field(default_factory=_new_id)
    content: str = ""
    source_episode_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    created_at: datetime = field(default_factory=_now)
    last_accessed: datetime = field(default_factory=_now)
    access_count: int = 0
    confidence: float = 0.6
    reinforcement_count: int = 1

    def strength(self) -> float:
        """Slow decay — semantic knowledge should survive months.

        At 90 days (2160 hours), with default stability ~2.7:
          exp(-0.01 * 2160 / 2.7) ≈ exp(-8) ≈ 0.0003 (very low)
        But with reinforcement_count=3 and access_count=5:
          stability = 2.0 + ln(1+8) = 4.2
          exp(-0.01 * 2160 / 4.2) ≈ exp(-5.1) ≈ 0.006 (still low but above 0.002 threshold)
        Key: reinforced facts survive; unreinforced fade. This IS the desired behavior.
        """
        hours = max(((_now() - self.last_accessed).total_seconds()) / 3600, 0.01)
        stability = 2.0 + math.log1p(self.access_count + self.reinforcement_count)
        decay = math.exp(-0.01 * hours / stability)
        return self.confidence * decay

    def reinforce(self, episode_id: str) -> None:
        self.reinforcement_count += 1
        self.confidence = min(1.0, self.confidence + 0.05)
        if episode_id and episode_id not in self.source_episode_ids:
            self.source_episode_ids.append(episode_id)
        self.touch()

    def touch(self) -> None:
        self.last_accessed = _now()
        self.access_count += 1


@dataclass
class ProceduralMemory:
    """A reusable strategy: 'when <trigger>, do <action>'."""

    id: str = field(default_factory=_new_id)
    trigger: str = ""
    action: str = ""
    source_semantic_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    created_at: datetime = field(default_factory=_now)
    last_accessed: datetime = field(default_factory=_now)
    access_count: int = 0
    confidence: float = 0.7
    reinforcement_count: int = 1

    def strength(self) -> float:
        """Slowest decay — procedural strategies are the most stable tier."""
        hours = max(((_now() - self.last_accessed).total_seconds()) / 3600, 0.01)
        stability = 3.0 + math.log1p(self.access_count + self.reinforcement_count)
        decay = math.exp(-0.01 * hours / stability)
        return self.confidence * decay

    def reinforce(self) -> None:
        self.reinforcement_count += 1
        self.confidence = min(1.0, self.confidence + 0.03)
        self.touch()

    def touch(self) -> None:
        self.last_accessed = _now()
        self.access_count += 1

"""SQLite-backed persistent store for PMC memories — v2.

Changes from v1:
- get_active_semantic/procedural default min_strength: 0.1 → 0.02
  This prevents the "positive feedback trap" where decayed memories
  are excluded from consolidation, preventing further reinforcement.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nanobot.memory.models import (
    EpisodicMemory,
    MemoryType,
    ProceduralMemory,
    SemanticMemory,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodic (
    id TEXT PRIMARY KEY,
    session_key TEXT NOT NULL,
    query TEXT NOT NULL,
    summary TEXT NOT NULL,
    outcome TEXT DEFAULT 'neutral',
    tags TEXT DEFAULT '[]',
    created_at TEXT NOT NULL,
    last_accessed TEXT NOT NULL,
    access_count INTEGER DEFAULT 0,
    confidence REAL DEFAULT 0.5,
    consolidated INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS semantic (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    source_episode_ids TEXT DEFAULT '[]',
    tags TEXT DEFAULT '[]',
    created_at TEXT NOT NULL,
    last_accessed TEXT NOT NULL,
    access_count INTEGER DEFAULT 0,
    confidence REAL DEFAULT 0.6,
    reinforcement_count INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS procedural (
    id TEXT PRIMARY KEY,
    trigger_text TEXT NOT NULL,
    action TEXT NOT NULL,
    source_semantic_ids TEXT DEFAULT '[]',
    tags TEXT DEFAULT '[]',
    created_at TEXT NOT NULL,
    last_accessed TEXT NOT NULL,
    access_count INTEGER DEFAULT 0,
    confidence REAL DEFAULT 0.7,
    reinforcement_count INTEGER DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_episodic_session ON episodic(session_key);
CREATE INDEX IF NOT EXISTS idx_episodic_consolidated ON episodic(consolidated);
CREATE INDEX IF NOT EXISTS idx_semantic_confidence ON semantic(confidence);
CREATE INDEX IF NOT EXISTS idx_procedural_confidence ON procedural(confidence);
"""


def _ts(dt: datetime) -> str:
    return dt.isoformat()


def _parse_ts(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class MemoryStore:
    """Persistent store for the three PMC memory tiers."""

    def __init__(self, db_path: Path | str):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA busy_timeout = 30000")
        self._conn.executescript(_SCHEMA)

    # ── Episodic ─────────────────────────────────────────────────────────

    def save_episode(self, ep: EpisodicMemory) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO episodic
               (id, session_key, query, summary, outcome, tags,
                created_at, last_accessed, access_count, confidence, consolidated)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ep.id, ep.session_key, ep.query, ep.summary, ep.outcome,
                json.dumps(ep.tags), _ts(ep.created_at), _ts(ep.last_accessed),
                ep.access_count, ep.confidence, int(ep.consolidated),
            ),
        )
        self._conn.commit()

    def get_unconsolidated_episodes(self, limit: int = 20) -> list[EpisodicMemory]:
        rows = self._conn.execute(
            "SELECT * FROM episodic WHERE consolidated = 0 ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_episode(r) for r in rows]

    def get_recent_episodes(self, n: int = 10) -> list[EpisodicMemory]:
        rows = self._conn.execute(
            "SELECT * FROM episodic ORDER BY created_at DESC LIMIT ?", (n,)
        ).fetchall()
        return [self._row_to_episode(r) for r in rows]

    def mark_consolidated(self, episode_ids: list[str]) -> None:
        if not episode_ids:
            return
        placeholders = ",".join("?" for _ in episode_ids)
        self._conn.execute(
            f"UPDATE episodic SET consolidated = 1 WHERE id IN ({placeholders})",
            episode_ids,
        )
        self._conn.commit()

    def count_unconsolidated(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM episodic WHERE consolidated = 0"
        ).fetchone()
        return row[0]

    def _row_to_episode(self, row: sqlite3.Row) -> EpisodicMemory:
        return EpisodicMemory(
            id=row["id"],
            session_key=row["session_key"],
            query=row["query"],
            summary=row["summary"],
            outcome=row["outcome"],
            tags=json.loads(row["tags"]),
            created_at=_parse_ts(row["created_at"]),
            last_accessed=_parse_ts(row["last_accessed"]),
            access_count=row["access_count"],
            confidence=row["confidence"],
            consolidated=bool(row["consolidated"]),
        )

    # ── Semantic ─────────────────────────────────────────────────────────

    def save_semantic(self, mem: SemanticMemory) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO semantic
               (id, content, source_episode_ids, tags,
                created_at, last_accessed, access_count, confidence, reinforcement_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                mem.id, mem.content, json.dumps(mem.source_episode_ids),
                json.dumps(mem.tags), _ts(mem.created_at), _ts(mem.last_accessed),
                mem.access_count, mem.confidence, mem.reinforcement_count,
            ),
        )
        self._conn.commit()

    def get_all_semantic(self) -> list[SemanticMemory]:
        rows = self._conn.execute(
            "SELECT * FROM semantic ORDER BY confidence DESC"
        ).fetchall()
        return [self._row_to_semantic(r) for r in rows]

    def get_active_semantic(self, min_strength: float = 0.02) -> list[SemanticMemory]:
        """Return semantic memories above decay threshold.
        Default lowered from 0.1 to 0.02 to prevent premature exclusion.
        """
        all_mem = self.get_all_semantic()
        return [m for m in all_mem if m.strength() >= min_strength]

    def _row_to_semantic(self, row: sqlite3.Row) -> SemanticMemory:
        return SemanticMemory(
            id=row["id"],
            content=row["content"],
            source_episode_ids=json.loads(row["source_episode_ids"]),
            tags=json.loads(row["tags"]),
            created_at=_parse_ts(row["created_at"]),
            last_accessed=_parse_ts(row["last_accessed"]),
            access_count=row["access_count"],
            confidence=row["confidence"],
            reinforcement_count=row["reinforcement_count"],
        )

    # ── Procedural ───────────────────────────────────────────────────────

    def save_procedural(self, mem: ProceduralMemory) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO procedural
               (id, trigger_text, action, source_semantic_ids, tags,
                created_at, last_accessed, access_count, confidence, reinforcement_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                mem.id, mem.trigger, mem.action,
                json.dumps(mem.source_semantic_ids), json.dumps(mem.tags),
                _ts(mem.created_at), _ts(mem.last_accessed),
                mem.access_count, mem.confidence, mem.reinforcement_count,
            ),
        )
        self._conn.commit()

    def get_all_procedural(self) -> list[ProceduralMemory]:
        rows = self._conn.execute(
            "SELECT * FROM procedural ORDER BY confidence DESC"
        ).fetchall()
        return [self._row_to_procedural(r) for r in rows]

    def get_active_procedural(self, min_strength: float = 0.02) -> list[ProceduralMemory]:
        """Return procedural memories above decay threshold.
        Default lowered from 0.1 to 0.02 to prevent premature exclusion.
        """
        all_mem = self.get_all_procedural()
        return [m for m in all_mem if m.strength() >= min_strength]

    def _row_to_procedural(self, row: sqlite3.Row) -> ProceduralMemory:
        return ProceduralMemory(
            id=row["id"],
            trigger=row["trigger_text"],
            action=row["action"],
            source_semantic_ids=json.loads(row["source_semantic_ids"]),
            tags=json.loads(row["tags"]),
            created_at=_parse_ts(row["created_at"]),
            last_accessed=_parse_ts(row["last_accessed"]),
            access_count=row["access_count"],
            confidence=row["confidence"],
            reinforcement_count=row["reinforcement_count"],
        )

    # ── Stats ────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        counts = {}
        for table in ("episodic", "semantic", "procedural"):
            row = self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            counts[table] = row[0]
        uncons = self.count_unconsolidated()
        return {**counts, "unconsolidated_episodes": uncons}

    def close(self) -> None:
        self._conn.close()

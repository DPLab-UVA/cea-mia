"""Agent interface for CEA-MI attack \u2014 interact with nanobot's memory-augmented LLM."""
from __future__ import annotations

import asyncio
import json
import shutil
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx

import sys

from config import (
    DEFAULT_API_BASE,
    DEFAULT_API_KEY,
    DEFAULT_MODEL,
    DEFAULT_NANOBOT_DB_PATH,
    DEFAULT_NANOBOT_PROJECT,
)

if str(DEFAULT_NANOBOT_PROJECT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_NANOBOT_PROJECT))

from nanobot.memory.store import MemoryStore
from nanobot.memory.recall import Recall, RecallResult, _tokenize, _relevance
from nanobot.memory.models import SemanticMemory, EpisodicMemory


class AgentInterface:
    """Unified interface to the nanobot agent for membership inference probing."""

    def __init__(
        self,
        api_base: str = DEFAULT_API_BASE,
        api_key: str = DEFAULT_API_KEY,
        model: str = DEFAULT_MODEL,
        db_path: Optional[Path] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.db_path = db_path or DEFAULT_NANOBOT_DB_PATH
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = httpx.AsyncClient(timeout=120)
        self.query_count = 0
        self._store: Optional[MemoryStore] = None
        self._recall: Optional[Recall] = None

    @property
    def store(self) -> MemoryStore:
        if self._store is None:
            self._store = MemoryStore(self.db_path)
            self._recall = Recall(self._store)
        return self._store

    @property
    def recall_engine(self) -> Recall:
        if self._recall is None:
            _ = self.store  # triggers lazy init
        return self._recall

    # \u2500\u2500 Memory manipulation (for experiment setup) \u2500\u2500

    def inject_semantic_memory(self, content: str, tags: list[str] = None) -> str:
        """Inject a fact directly into semantic memory. Returns memory ID."""
        mem = SemanticMemory(
            content=content,
            tags=tags or [],
            confidence=0.8,
            reinforcement_count=2,
        )
        self.store.save_semantic(mem)
        return mem.id

    def clear_all_memory(self):
        """Wipe all memory tables."""
        conn = sqlite3.connect(str(self.db_path))
        for table in ("episodic", "semantic", "procedural"):
            conn.execute(f"DELETE FROM {table}")
        conn.commit()
        conn.close()
        # Reset cached store/recall
        self._store = None
        self._recall = None

    def snapshot_memory(self, snapshot_path: Path) -> Path:
        """Copy the current memory DB to a snapshot file."""
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.db_path, snapshot_path)
        return snapshot_path

    def restore_memory(self, snapshot_path: Path):
        """Restore memory DB from a snapshot."""
        if self._store:
            self._store.close()
            self._store = None
            self._recall = None
        shutil.copy2(snapshot_path, self.db_path)

    def get_all_semantic_memories(self) -> list[SemanticMemory]:
        """White-box: get all semantic memories."""
        return self.store.get_all_semantic()

    def recall_for_query(self, query: str) -> RecallResult:
        """White-box: perform memory recall and return results."""
        return self.recall_engine.recall(query)

    def get_recall_scores(self, query: str) -> list[dict]:
        """White-box: get detailed recall scoring for a query."""
        query_tokens = _tokenize(query)
        results = []
        for s in self.store.get_all_semantic():
            rel = _relevance(query_tokens, s.content, s.tags)
            results.append({
                "memory_id": s.id,
                "content": s.content,
                "relevance": rel,
                "strength": s.strength(),
                "confidence": s.confidence,
            })
        results.sort(key=lambda x: (x["relevance"], x["strength"]), reverse=True)
        return results

    # \u2500\u2500 Build system prompt (replicating nanobot behavior) \u2500\u2500

    def _build_system_prompt(self, memory_section: str = "") -> str:
        import platform as plat
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        ws = str(Path.home() / ".nanobot" / "workspace")

        pmc_section = ""
        if memory_section:
            pmc_section = f"\n\n## Recalled Memories (PMC)\n\n{memory_section}"
            pmc_section += (
                "\n\nNote: These memories were automatically recalled based on your query. "
                "Procedural strategies are the most distilled and reliable. "
                "Semantic knowledge has been verified across multiple interactions. "
                "Episodic memories are raw recent experiences."
            )

        return f"""# nanobot

You are nanobot, a helpful AI assistant running locally.

## Current Time
{now}

## Workspace
{ws}

## Guidelines
- Be helpful, accurate, and concise
- Pay attention to recalled memories \u2014 they contain lessons from past interactions
{pmc_section}"""

    # \u2500\u2500 Query the agent \u2500\u2500

    async def query_blackbox(self, message: str, session_history: list[dict] = None) -> dict:
        """Black-box query: text response only."""
        recalled = self.recall_engine.recall(message)
        memory_prompt = recalled.format_for_prompt()

        messages = [{"role": "system", "content": self._build_system_prompt(memory_prompt)}]
        if session_history:
            messages.extend(session_history)
        messages.append({"role": "user", "content": message})

        start = time.monotonic()
        resp = await self.client.post(
            f"{self.api_base}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
        )
        latency = (time.monotonic() - start) * 1000
        resp.raise_for_status()
        data = resp.json()
        self.query_count += 1

        return {
            "response": data["choices"][0]["message"]["content"],
            "latency_ms": latency,
            "recall_triggered": not recalled.is_empty(),
        }

    async def query_graybox(self, message: str, session_history: list[dict] = None) -> dict:
        """Gray-box query: text + logprobs."""
        recalled = self.recall_engine.recall(message)
        memory_prompt = recalled.format_for_prompt()

        messages = [{"role": "system", "content": self._build_system_prompt(memory_prompt)}]
        if session_history:
            messages.extend(session_history)
        messages.append({"role": "user", "content": message})

        start = time.monotonic()
        resp = await self.client.post(
            f"{self.api_base}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "logprobs": True,
                "top_logprobs": 5,
            },
        )
        latency = (time.monotonic() - start) * 1000
        resp.raise_for_status()
        data = resp.json()
        self.query_count += 1

        choice = data["choices"][0]
        logprobs_data = choice.get("logprobs", {})
        token_logprobs = []
        if logprobs_data and logprobs_data.get("content"):
            token_logprobs = [t["logprob"] for t in logprobs_data["content"]]

        return {
            "response": choice["message"]["content"],
            "latency_ms": latency,
            "logprobs": token_logprobs,
            "mean_logprob": sum(token_logprobs) / len(token_logprobs) if token_logprobs else None,
            "recall_triggered": not recalled.is_empty(),
        }

    async def query_whitebox(self, message: str, session_history: list[dict] = None) -> dict:
        """White-box query: text + logprobs + memory recall details."""
        # Get detailed recall info first
        recall_scores = self.get_recall_scores(message)
        recalled = self.recall_engine.recall(message)
        memory_prompt = recalled.format_for_prompt()

        messages = [{"role": "system", "content": self._build_system_prompt(memory_prompt)}]
        if session_history:
            messages.extend(session_history)
        messages.append({"role": "user", "content": message})

        start = time.monotonic()
        resp = await self.client.post(
            f"{self.api_base}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "logprobs": True,
                "top_logprobs": 5,
            },
        )
        latency = (time.monotonic() - start) * 1000
        resp.raise_for_status()
        data = resp.json()
        self.query_count += 1

        choice = data["choices"][0]
        logprobs_data = choice.get("logprobs", {})
        token_logprobs = []
        if logprobs_data and logprobs_data.get("content"):
            token_logprobs = [t["logprob"] for t in logprobs_data["content"]]

        top_scores = recall_scores[:5] if recall_scores else []

        return {
            "response": choice["message"]["content"],
            "latency_ms": latency,
            "logprobs": token_logprobs,
            "mean_logprob": sum(token_logprobs) / len(token_logprobs) if token_logprobs else None,
            "recall_triggered": not recalled.is_empty(),
            "recall_hit_count": len(recalled.semantic) + len(recalled.episodic),
            "recall_top_similarity": top_scores[0]["relevance"] if top_scores else 0.0,
            "recall_scores": top_scores,
            "memory_stats": self.store.stats(),
        }

    async def query(self, message: str, access_level: str = "blackbox", session_history: list[dict] = None) -> dict:
        """Dispatch to appropriate query method."""
        if access_level == "whitebox":
            return await self.query_whitebox(message, session_history)
        elif access_level == "graybox":
            return await self.query_graybox(message, session_history)
        else:
            return await self.query_blackbox(message, session_history)

    async def close(self):
        await self.client.aclose()
        if self._store:
            self._store.close()

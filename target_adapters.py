"""Target-agent adapters for CEA-MI experiments.

Each adapter exposes the small interface the attack needs:
prepare one user's member memories, query the memory-augmented agent, and close
any resources. This keeps the CEA-MI algorithm independent from a particular
memory implementation.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Protocol

from agent_interface import AgentInterface
from config import Config
from memory_unit import MemoryUnit, UserMemorySet


REPO_ROOT = Path(__file__).resolve().parent
TARGET_CHOICES = ("nanobot", "mem0", "memgpt")


class TargetAgent(Protocol):
    target_name: str

    def prepare_user_memory(self, user_set: UserMemorySet) -> int:
        """Reset target memory and load this user's member memories."""

    async def query(self, message: str, access_level: str = "blackbox") -> dict:
        """Query the target agent and return normalized response fields."""

    async def close(self) -> None:
        """Release any target resources."""


def _unit_type(unit: MemoryUnit) -> str:
    return getattr(unit.perlt_type, "value", str(unit.perlt_type))


def _memory_tags(unit: MemoryUnit) -> list[str]:
    tags = [
        f"user:{unit.user_id}",
        f"type:{_unit_type(unit)}",
        f"source:{unit.source_key}",
    ]
    if unit.topic:
        tags.append(f"topic:{unit.topic}")
    return tags


def _memory_metadata(unit: MemoryUnit, is_member: bool = True) -> dict:
    return {
        "user_id": unit.user_id,
        "is_member": is_member,
        "type": _unit_type(unit),
        "source": unit.source_key,
        "topic": unit.topic,
        "key_value": unit.key_value,
    }


class NanobotTarget:
    target_name = "nanobot"

    def __init__(self, cfg: Config, db_path: Optional[Path] = None):
        self.agent = AgentInterface(
            api_base=cfg.api_base,
            api_key=cfg.api_key,
            model=cfg.model,
            db_path=db_path or cfg.nanobot_db_path,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )

    def prepare_user_memory(self, user_set: UserMemorySet) -> int:
        self.agent.clear_all_memory()
        for unit in user_set.members:
            self.agent.inject_semantic_memory(unit.content, tags=_memory_tags(unit))
        return len(user_set.members)

    async def query(self, message: str, access_level: str = "blackbox") -> dict:
        return await self.agent.query(message, access_level=access_level)

    async def close(self) -> None:
        await self.agent.close()


class _EmbeddingMemoryTarget:
    """Shared adapter for lightweight Mem0/MemGPT-style embedding targets."""

    target_name = "embedding"

    def __init__(self, cfg: Config, memory_file: Path):
        self.cfg = cfg
        self.memory_file = Path(memory_file)
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        self.agent = self._create_agent()

    def _create_agent(self):
        raise NotImplementedError

    def prepare_user_memory(self, user_set: UserMemorySet) -> int:
        self.agent.memories = []
        for unit in user_set.members:
            self.agent.add_memory(unit.content, metadata=_memory_metadata(unit))
        self.agent.save()
        return len(user_set.members)

    @staticmethod
    def _serialize_recalled(recalled: list[dict]) -> list[dict]:
        serialized = []
        for item in recalled:
            if not isinstance(item, dict):
                continue
            similarity = float(item.get("similarity", 0.0) or 0.0)
            serialized.append({
                "type": "semantic",
                "content": item.get("content", ""),
                "similarity": similarity,
                "relevance": similarity,
                "metadata": item.get("metadata", {}),
            })
        return serialized

    async def query(self, message: str, access_level: str = "blackbox") -> dict:
        result = await self.agent.query(message, access_level=access_level)

        if access_level == "whitebox":
            recalled = self._serialize_recalled(self.agent.recall(message))
            result.setdefault("recalled_memories", recalled)
            result.setdefault("recall_scores", recalled[:5])
            result.setdefault("recall_hit_count", len(recalled))
            result.setdefault(
                "recall_top_similarity",
                recalled[0]["similarity"] if recalled else 0.0,
            )
            result.setdefault("recall_triggered", bool(recalled))
            result.setdefault("memory_stats", {
                "target": self.target_name,
                "memory_count": len(self.agent.memories),
                "memory_file": str(self.memory_file),
            })

        return result

    async def close(self) -> None:
        return None


class Mem0Target(_EmbeddingMemoryTarget):
    target_name = "mem0"

    def _create_agent(self):
        target_dir = REPO_ROOT / "mem0_target"
        if str(target_dir) not in sys.path:
            sys.path.insert(0, str(target_dir))
        from setup_mem0 import Mem0Agent

        return Mem0Agent(
            db_path=self.memory_file,
            vllm_base=self.cfg.api_base,
            vllm_model=self.cfg.model,
            vllm_api_key=self.cfg.api_key,
        )


class MemGPTTarget(_EmbeddingMemoryTarget):
    target_name = "memgpt"

    def _create_agent(self):
        target_dir = REPO_ROOT / "memgpt_target"
        if str(target_dir) not in sys.path:
            sys.path.insert(0, str(target_dir))
        from setup_memgpt import EmbeddingMemoryAgent

        return EmbeddingMemoryAgent(
            db_path=self.memory_file,
            vllm_base=self.cfg.api_base,
            vllm_model=self.cfg.model,
            vllm_api_key=self.cfg.api_key,
        )


def load_target_agent(
    target: str,
    cfg: Config,
    *,
    db_path: Optional[Path] = None,
    memory_file: Optional[Path] = None,
) -> TargetAgent:
    """Create a target adapter by name."""
    if target == "nanobot":
        return NanobotTarget(cfg, db_path=db_path)

    if target == "mem0":
        if memory_file is None:
            raise ValueError("mem0 target requires a memory_file working path")
        return Mem0Target(cfg, Path(memory_file))

    if target == "memgpt":
        if memory_file is None:
            raise ValueError("memgpt target requires a memory_file working path")
        return MemGPTTarget(cfg, Path(memory_file))

    raise ValueError(f"Unknown target: {target}")

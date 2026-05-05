"""Progressive Memory Consolidation engine — v3 (all patches applied).

Changes from v2:
- Patch 2: mark_consolidated ONLY when something was learned or reinforced
- Patch 3: _new_semantic_since_proc ONLY resets on successful procedural output
- Patch 4: Explicit min_strength=0.02 in consolidator's get_active calls
- Maintained: improved prompts with personal fact extraction + code-level dedup
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

from nanobot.memory.models import (
    EpisodicMemory,
    ProceduralMemory,
    SemanticMemory,
)
from nanobot.memory.store import MemoryStore

logger = logging.getLogger(__name__)

# ── Prompts ──────────────────────────────────────────────────────────────────

CONSOLIDATE_EPISODIC_PROMPT = """\
You are a memory consolidation system for a personal AI assistant.
Given a batch of recent interaction episodes, extract **factual knowledge** about the user and their world.

Episodes:
{episodes}

Existing semantic memories (DO NOT duplicate these — reinforce if overlapping):
{existing_semantic}

CRITICAL INSTRUCTIONS:
- Extract TWO types of knowledge:
  1. **User profile facts**: name, preferences, habits, relationships, pets, locations, schedules, tools they use, etc.
     Examples: "The user's name is Alex", "The user has a cat named Mochi (Scottish Fold, 3 years old)", "The user's partner is Jamie"
  2. **Technical/domain knowledge**: reusable insights from coding or research discussions.
     Examples: "The vit-robust project uses PyTorch with src layout", "PGD delta tensor must be recreated per batch"

- PRIORITIZE user profile facts — these are the most valuable for personalization.
- Use SPECIFIC names, numbers, and details — never write "the user's pet" when you know it's "Mochi".
- Each fact must be a standalone statement that makes sense without context.
- If an episode confirms an existing semantic memory, add its ID to reinforce_semantic_ids.
- Extract 1-5 facts. Quality over quantity.

Respond in JSON ONLY (no markdown fences, no extra text):
{{"new_semantic": [{{"content": "...", "tags": ["tag1", "tag2"], "source_episode_ids": ["id1"]}}], "reinforce_semantic_ids": ["existing_id"], "reinforcing_episode_ids": ["episode_id"]}}

If nothing new can be extracted, return {{"new_semantic": [], "reinforce_semantic_ids": [], "reinforcing_episode_ids": []}}
"""

CONSOLIDATE_SEMANTIC_PROMPT = """\
You are a strategy extraction system for a personal AI assistant.
Given factual knowledge (semantic memories), extract reusable **action strategies**.

NEW semantic memories to process:
{semantic_memories}

ALL existing procedural strategies (you MUST NOT create duplicates of these):
{existing_procedural}

CRITICAL INSTRUCTIONS:
- Extract strategies ONLY if genuinely NEW and NOT covered by existing strategies above.
- If an existing strategy already covers the same idea (even different wording), add its ID to reinforce_procedural_ids.
- Two types:
  1. **User preference strategies**: "When the user asks for food -> remember they love ramen and go to Tatsu-Ya South"
  2. **Technical strategies**: "When debugging identical outputs in iterative computation -> check for mutable state reuse"
- Maximum 1-2 NEW strategies. Return EMPTY new_procedural if existing ones already cover it.
- It is BETTER to reinforce than to create a near-duplicate.

Respond in JSON ONLY:
{{"new_procedural": [{{"trigger": "When ...", "action": "Do ...", "tags": ["tag1"], "source_semantic_ids": ["id1"]}}], "reinforce_procedural_ids": ["existing_id"]}}

If no NEW strategies needed, return {{"new_procedural": [], "reinforce_procedural_ids": []}}
"""


class LLMCallable(Protocol):
    async def __call__(self, messages: list[dict[str, Any]]) -> str: ...


def _is_duplicate_semantic(new_content: str, existing: list[SemanticMemory], threshold: float = 0.6) -> str | None:
    new_tokens = set(new_content.lower().split())
    for mem in existing:
        existing_tokens = set(mem.content.lower().split())
        if not new_tokens or not existing_tokens:
            continue
        overlap = len(new_tokens & existing_tokens)
        similarity = overlap / min(len(new_tokens), len(existing_tokens))
        if similarity > threshold:
            return mem.id
    return None


def _is_duplicate_procedural(new_trigger: str, new_action: str, existing: list[ProceduralMemory], threshold: float = 0.5) -> str | None:
    new_tokens = set(f"{new_trigger} {new_action}".lower().split())
    for proc in existing:
        existing_tokens = set(f"{proc.trigger} {proc.action}".lower().split())
        if not new_tokens or not existing_tokens:
            continue
        overlap = len(new_tokens & existing_tokens)
        similarity = overlap / min(len(new_tokens), len(existing_tokens))
        if similarity > threshold:
            return proc.id
    return None


class Consolidator:
    def __init__(
        self,
        store: MemoryStore,
        llm_call: LLMCallable,
        episode_batch_size: int = 5,
        semantic_threshold: int = 6,
    ):
        self.store = store
        self.llm_call = llm_call
        self.episode_batch_size = episode_batch_size
        self.semantic_threshold = semantic_threshold
        self._new_semantic_since_proc = 0

    async def maybe_consolidate(self) -> dict[str, int]:
        result = {"new_semantic": 0, "new_procedural": 0, "reinforced": 0}

        # Phase 1: Episodic -> Semantic
        n_uncons = self.store.count_unconsolidated()
        if n_uncons >= self.episode_batch_size:
            r1 = await self._consolidate_episodes()
            result["new_semantic"] = r1.get("new_semantic", 0)
            result["reinforced"] += r1.get("reinforced", 0)
            self._new_semantic_since_proc += r1.get("new_semantic", 0)

        # Phase 2: Semantic -> Procedural
        # PATCH 3: Only reset counter on SUCCESS
        if self._new_semantic_since_proc >= self.semantic_threshold:
            r2 = await self._consolidate_semantic()
            result["new_procedural"] = r2.get("new_procedural", 0)
            result["reinforced"] += r2.get("reinforced", 0)
            # Only reset if something was actually produced or reinforced
            if r2.get("new_procedural", 0) > 0 or r2.get("reinforced", 0) > 0:
                self._new_semantic_since_proc = 0

        return result

    async def _consolidate_episodes(self) -> dict[str, int]:
        episodes = self.store.get_unconsolidated_episodes(limit=self.episode_batch_size)
        if not episodes:
            return {}

        existing = self.store.get_all_semantic()

        episodes_text = "\n".join(
            f"- [{ep.id}] (outcome={ep.outcome}, tags={ep.tags}) "
            f"Query: {ep.query[:150]} | Summary: {ep.summary[:200]}"
            for ep in episodes
        )
        existing_text = "\n".join(
            f"- [{m.id}] (confidence={m.confidence:.2f}, reinforced={m.reinforcement_count}x) "
            f"{m.content[:150]}"
            for m in existing[:15]
        ) or "(none yet)"

        prompt = CONSOLIDATE_EPISODIC_PROMPT.format(
            episodes=episodes_text, existing_semantic=existing_text
        )

        try:
            raw = await self.llm_call([
                {"role": "system", "content": "You are a memory consolidation engine. Respond only in valid JSON."},
                {"role": "user", "content": prompt},
            ])
            data = _parse_json(raw)
        except Exception as e:
            logger.warning(f"Episodic consolidation failed: {e}")
            return {}

        counts = {"new_semantic": 0, "reinforced": 0}

        for item in data.get("new_semantic", []):
            content = item.get("content", "").strip()
            if not content:
                continue

            dup_id = _is_duplicate_semantic(content, existing)
            if dup_id:
                existing_map = {m.id: m for m in existing}
                if dup_id in existing_map:
                    m = existing_map[dup_id]
                    source_ids = item.get("source_episode_ids", [])
                    m.reinforce(source_ids[0] if source_ids else "")
                    self.store.save_semantic(m)
                    counts["reinforced"] += 1
                continue

            mem = SemanticMemory(
                content=content,
                source_episode_ids=item.get("source_episode_ids", []),
                tags=item.get("tags", []),
            )
            self.store.save_semantic(mem)
            existing.append(mem)
            counts["new_semantic"] += 1

        existing_map = {m.id: m for m in existing}
        for sid in data.get("reinforce_semantic_ids", []):
            if sid in existing_map:
                m = existing_map[sid]
                eids = data.get("reinforcing_episode_ids", [])
                m.reinforce(eids[0] if eids else "")
                self.store.save_semantic(m)
                counts["reinforced"] += 1

        # PATCH 2: Only mark consolidated if something was learned or reinforced
        if counts.get("new_semantic", 0) > 0 or counts.get("reinforced", 0) > 0:
            self.store.mark_consolidated([ep.id for ep in episodes])
        # else: leave them unconsolidated for retry next time

        return counts

    async def _consolidate_semantic(self) -> dict[str, int]:
        # PATCH 4: Explicit min_strength to prevent silent threshold mismatch
        semantic = self.store.get_active_semantic(min_strength=0.02)
        if len(semantic) < self.semantic_threshold:
            return {}

        existing_proc = self.store.get_all_procedural()

        recent_semantic = sorted(semantic, key=lambda s: s.created_at, reverse=True)[:self.semantic_threshold]

        semantic_text = "\n".join(
            f"- [{m.id}] (confidence={m.confidence:.2f}, tags={m.tags}) {m.content[:200]}"
            for m in recent_semantic
        )

        proc_text = "\n".join(
            f"- [{p.id}] When: {p.trigger[:100]} -> Do: {p.action[:100]}"
            for p in existing_proc[:20]
        ) or "(none yet)"

        prompt = CONSOLIDATE_SEMANTIC_PROMPT.format(
            semantic_memories=semantic_text, existing_procedural=proc_text
        )

        try:
            raw = await self.llm_call([
                {"role": "system", "content": "You are a strategy extraction engine. Respond only in valid JSON."},
                {"role": "user", "content": prompt},
            ])
            data = _parse_json(raw)
        except Exception as e:
            logger.warning(f"Semantic consolidation failed: {e}")
            return {}

        counts = {"new_procedural": 0, "reinforced": 0}

        for item in data.get("new_procedural", []):
            trigger = item.get("trigger", "").strip()
            action = item.get("action", "").strip()
            if not trigger or not action:
                continue

            dup_id = _is_duplicate_procedural(trigger, action, existing_proc)
            if dup_id:
                proc_map = {p.id: p for p in existing_proc}
                if dup_id in proc_map:
                    proc_map[dup_id].reinforce()
                    self.store.save_procedural(proc_map[dup_id])
                    counts["reinforced"] += 1
                continue

            mem = ProceduralMemory(
                trigger=trigger,
                action=action,
                source_semantic_ids=item.get("source_semantic_ids", []),
                tags=item.get("tags", []),
            )
            self.store.save_procedural(mem)
            existing_proc.append(mem)
            counts["new_procedural"] += 1

        proc_map = {p.id: p for p in existing_proc}
        for pid in data.get("reinforce_procedural_ids", []):
            if pid in proc_map:
                proc_map[pid].reinforce()
                self.store.save_procedural(proc_map[pid])
                counts["reinforced"] += 1

        return counts


def _parse_json(raw: str) -> dict:
    """Robustly parse JSON from LLM output."""
    text = raw.strip()
    # Remove markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    # Find outermost { }
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        text = text[start:end]
    return json.loads(text)

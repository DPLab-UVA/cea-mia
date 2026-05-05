"""Decoy builder — construct counterfactual facts for contrastive probing.

Two implementations:
- LegacyDecoyBuilder: Original implementation using hardcoded patterns (deprecated)
- LLMDecoyBuilder: New implementation using vLLM to generate high-quality counterparts
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional, Union

import httpx

from config import DEFAULT_API_BASE, DEFAULT_API_KEY, DEFAULT_MODEL
from http_utils import post_with_retry
from json_utils import (
    JsonObjectError,
    extract_json_object_text,
    json_error_snippet,
    loads_json_object,
)
from models import Fact, DecoyPair
from memory_unit import MemoryUnit, MemoryUnitPair, PerltType

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# LLM-based Decoy Builder (New)
# ═══════════════════════════════════════════════════════════════════════════

COUNTERFACTUAL_PROMPT = """\
Create a counterfactual version of the personal memory.

A valid counterfactual changes one key value so that the original and new
statement cannot both be true in the same context. Keep the new statement
equally plausible and natural.

## Requirements:
1. **Preserve context**: Keep the same subject, context, sentence shape, length, and style.
2. **Mutual exclusion**: Replace one core value with a plausible mutually exclusive value in the same topic/domain.
3. **No compatible values**: Do not use a value already stated, implied, synonymous, broader/narrower, entailed, or otherwise compatible with the original.
4. **No negation-only edits**: Do not use negation-only edits.

## Examples:
Original: "The user's favorite color is blue."
Good: "The user's favorite color is green."

Original: "I had lunch with my colleague Sarah at the Italian restaurant downtown yesterday."
Good: "I had lunch with my colleague Mike at the Thai restaurant near the office yesterday."

Original: "Wang Xiaoming is interested in photography and basketball."
Good: "Wang Xiaoming is interested in painting and tennis."
Bad: "Wang Xiaoming is interested in basketball." (basketball is already true in the original)

## Input:
{content}

## Output:
Return ONLY the alternative statement, nothing else. No quotes, no explanation."""


HARD_NEGATIVE_PROMPT = """\
Given this fact about a user, generate a DIFFERENT fact about the same general topic domain.

The new fact should:
1. Be about the same general domain (if food → different food fact; if work → different work fact)
2. Contain ENTIRELY different specific information
3. Be plausible as a real memory

## Examples:
Original: "The user prefers black coffee."
Different fact: "The user drinks two cups of tea every morning."

Original: "The training cluster uses RTX 3090 GPUs."
Different fact: "The training jobs are scheduled using Slurm."

## Input:
{content}

## Output:
Return ONLY the new fact, nothing else."""

MEMORY_PAIR_PROMPT = """\
You are an expert at creating paired memories for contrastive analysis. Given a personal memory, \
design a counterfactual decoy version and extract key metadata for structured comparison.
The decoy should differ on key value while keeping the same context, and it should not be true at the same time as the original.

## Key Concept:
- **topic**: A single atomic dimension/axis that can be varied (e.g., "hobby", "favorite color", "job title").
- **original_key_value**: The value in the original memory for that topic.
- **decoy_key_value**: A plausible mutually exclusive value for the same topic.
- **decoy_content**: A counterfactual statement made by replacing only the key_value; keep the same subject, context, sentence structure, and other details. The original and decoy should not both be true in the same context.

## Requirements:
1. **Atomic topic**: Vary one atomic topic; prefer attributes over names when possible.
2. **Mutual exclusion**: The decoy value must conflict with the original value on that topic.
3. **No semantic overlap**: Do not use values already stated/implied by the original, synonyms, paraphrases, broader/narrower values, entailed values, or compatible restatements.
4. **Multi-value handling**: For multi-value memories, replace the whole value set or choose a different single-valued topic; never use one existing value as the decoy.
5. **Natural minimal edit**: Avoid negation-only edits and keep the decoy natural.

## Good examples:
Input: "Wang Xiaoming is interested in photography."
Output:
{{
  "topic": "hobby",
  "original_key_value": "photography",
  "decoy_key_value": "painting",
  "decoy_content": "Wang Xiaoming is interested in painting."
}}

Input: "Wang Xiaoming is interested in photography and basketball."
Output:
{{
  "topic": "hobbies",
  "original_key_value": "photography and basketball",
  "decoy_key_value": "painting and tennis",
  "decoy_content": "Wang Xiaoming is interested in painting and tennis."
}}

Input: "Wang Xiaoming uses a smart office solution called iConnect."
Output:
{{
  "topic": "office tool",
  "original_key_value": "iConnect",
  "decoy_key_value": "WorkFlow Pro",
  "decoy_content": "Wang Xiaoming uses a smart office solution called WorkFlow Pro."
}}

Bad examples (avoid these):
- topic="name", decoy="Li Xiaohong is interested in photography" for "Wang Xiaoming is interested in photography" (changed the subject, not the attribute)
- decoy_key_value="basketball" for "photography and basketball" (already true)
- decoy_key_value="coordination and planning" for "organizing and arranging" (paraphrase)
- decoy_key_value="concerned about success" for "supports him and is proud" (compatible/entailed)

## Input:
{content}

## Output:
Return ONLY valid JSON in this exact shape, nothing else:
{{
  "topic": "single atomic topic/axis (prefer attribute over subject name)",
  "original_key_value": "the single value for that topic",
  "decoy_key_value": "mutually exclusive plausible value for the same topic",
  "decoy_content": "full sentence with ONLY key_value changed"
}}
"""


class LLMDecoyBuilder:
    """Construct high-quality decoy facts using LLM.

    For each candidate memory/fact f, builds f⁻ that:
    - Has the same topic, structure, and length
    - Replaces the key value with a plausible alternative
    - Is equally "natural" and "answerable" (critical for avoiding bias)

    Supports both Fact (from models.py) and MemoryUnit (from memory_unit.py).
    """

    def __init__(
        self,
        api_base: str = DEFAULT_API_BASE,
        api_key: str = DEFAULT_API_KEY,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        max_retries: int = 2,
    ):
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    @staticmethod
    def _extract_json_text(raw: str) -> str:
        """Backward-compatible wrapper around the shared JSON extractor."""
        return extract_json_object_text(raw)

    @staticmethod
    def _json_error_snippet(text: str, pos: int, window: int = 80) -> str:
        """Backward-compatible wrapper around the shared JSON error snippet."""
        return json_error_snippet(text, pos, window=window)

    async def _call_llm(self, prompt: str, temperature: Optional[float] = None) -> str:
        """Call vLLM API to generate text with retry on transient failures."""
        temp = temperature if temperature is not None else self.temperature
        resp = await post_with_retry(
            self.client,
            f"{self.api_base}/chat/completions",
            max_retries=self.max_retries,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temp,
                "max_tokens": 256,
            },
        )
        content = resp.json()["choices"][0]["message"]["content"]
        return content.strip().strip('"').strip("'")

    async def _call_llm_json(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        required_keys: tuple[str, ...] = (),
    ) -> dict:
        """Call vLLM and parse a JSON object response, retrying malformed JSON."""
        attempts = max(1, self.max_retries + 1)
        last_error: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            call_temperature = temperature if attempt == 1 else 0.0
            raw = await self._call_llm(prompt, call_temperature)
            try:
                return loads_json_object(raw, required_keys=required_keys)
            except JsonObjectError as exc:
                last_error = exc
                log.warning(
                    "LLM returned invalid JSON for decoy pair "
                    "(attempt %d/%d): %s | raw=%r",
                    attempt,
                    attempts,
                    exc,
                    raw[:500].replace("\n", "\\n"),
                )
        assert last_error is not None
        raise last_error

    # ── Building counterfactual decoys ─────────────────────────────────────

    async def build_counterfactual(
        self, content: str, temperature: Optional[float] = None
    ) -> str:
        """Generate a counterfactual version of the given content.

        Args:
            content: The original fact/memory content
            temperature: Optional override for generation temperature

        Returns:
            The counterfactual content string
        """
        prompt = COUNTERFACTUAL_PROMPT.format(content=content)
        return await self._call_llm(prompt, temperature)

    # ── MemoryUnit support ─────────────────────────────────────────────────

    async def build_decoy_for_memory(self, unit: MemoryUnit) -> MemoryUnit:
        """Build a counterfactual decoy for a MemoryUnit."""
        pair = await self.build_decoy_pair_for_memory(unit)
        return pair.decoy

    async def build_decoy_pair_for_memory(self, unit: MemoryUnit) -> MemoryUnitPair:
        """Build a paired original/decoy MemoryUnit with shared topic metadata."""
        prompt = MEMORY_PAIR_PROMPT.format(
            content=unit.content,
        )
        required_keys = ("topic", "original_key_value", "decoy_key_value", "decoy_content")
        attempts = max(1, self.max_retries + 1)
        last_error: Optional[Exception] = None

        for attempt in range(1, attempts + 1):
            try:
                payload = await self._call_llm_json(
                    prompt,
                    temperature=self.temperature if attempt == 1 else 0.0,
                    required_keys=required_keys,
                )
                return self._memory_pair_from_payload(unit, payload)
            except Exception as exc:
                last_error = exc
                log.warning(
                    "Invalid decoy pair for unit %s (attempt %d/%d): %s | payload_source=%r",
                    unit.id,
                    attempt,
                    attempts,
                    exc,
                    unit.content[:180].replace("\n", " "),
                )

        raise ValueError(f"Failed to build valid decoy pair after {attempts} attempts") from last_error

    @staticmethod
    def _memory_pair_from_payload(unit: MemoryUnit, payload: dict) -> MemoryUnitPair:
        topic = (payload.get("topic") or unit.topic or "").strip() or None
        original_key_value = (
            payload.get("original_key_value")
            or payload.get("key_value")
            or unit.key_value
            or ""
        ).strip() or None
        decoy_key_value = (payload.get("decoy_key_value") or "").strip() or None
        decoy_content = (payload.get("decoy_content") or "").strip()

        if not decoy_content:
            raise ValueError("LLM did not return non-empty decoy_content for memory pair")
        if not decoy_key_value:
            raise ValueError("LLM did not return non-empty decoy_key_value for memory pair")
        if not original_key_value:
            raise ValueError("LLM did not return non-empty original_key_value for memory pair")
        if decoy_key_value.lower() == original_key_value.lower():
            raise ValueError("LLM returned an unchanged decoy_key_value for memory pair")

        original = MemoryUnit(
            id=unit.id,
            content=unit.content,
            perlt_type=unit.perlt_type,
            topic=topic,
            user_id=unit.user_id,
            source_key=unit.source_key,
            is_member=unit.is_member,
            key_value=original_key_value,
            created_at=unit.created_at,
        )
        decoy = MemoryUnit(
            id=unit.id + "_decoy",
            content=decoy_content,
            perlt_type=unit.perlt_type,
            topic=topic,
            user_id=unit.user_id,
            source_key=unit.source_key,
            is_member=False,
            key_value=decoy_key_value,
            created_at=unit.created_at,
        )
        return MemoryUnitPair(original=original, decoy=decoy)

    # ── Fact (legacy model) support ────────────────────────────────────────

    async def build_decoy_for_fact(self, fact: Fact) -> Fact:
        """Build a counterfactual decoy for a Fact (legacy model)."""
        decoy_content = await self.build_counterfactual(fact.content)

        return Fact(
            content=decoy_content,
            topic=fact.topic + "_llm_decoy",
            key_value="[llm_generated]",
            category=fact.category,
            is_member=False,
        )

    async def build_decoy_pair_for_fact(self, fact: Fact) -> DecoyPair:
        """Build a counterfactual decoy pair for a Fact."""
        decoy = await self.build_decoy_for_fact(fact)
        return DecoyPair(fact=fact, decoy=decoy)

    # ── Batch processing ───────────────────────────────────────────────────

    async def build_decoys_batch(
        self,
        items: list[Union[Fact, MemoryUnit]],
        max_concurrency: int = 10,
    ) -> list[tuple | MemoryUnitPair]:
        """Build decoys for a batch of items with concurrency control.

        Args:
            items: List of Fact or MemoryUnit objects
            max_concurrency: Maximum parallel LLM calls

        Returns:
            List of decoy pairs. MemoryUnit inputs return MemoryUnitPair objects.
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def process_one(item):
            async with semaphore:
                try:
                    if isinstance(item, MemoryUnit):
                        return await self.build_decoy_pair_for_memory(item)
                    else:
                        pair = await self.build_decoy_pair_for_fact(item)
                        return (pair.fact, pair.decoy)
                except Exception as e:
                    log.warning("Failed to build decoy for %s: %s", item.content[:50], e)
                    return None

        results = await asyncio.gather(*[process_one(item) for item in items])
        return [r for r in results if r is not None]


# ═══════════════════════════════════════════════════════════════════════════
# Legacy Decoy Builder (Deprecated)
# ═══════════════════════════════════════════════════════════════════════════

class LegacyDecoyBuilder:
    """[DEPRECATED] Original DecoyBuilder using hardcoded LLM prompts.

    Use LLMDecoyBuilder instead for better quality counterfactuals.

    This class is kept for backward compatibility.
    """

    def __init__(
        self,
        api_base: str = DEFAULT_API_BASE,
        api_key: str = DEFAULT_API_KEY,
        model: str = DEFAULT_MODEL,
    ):
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.client = httpx.AsyncClient(timeout=60)

    async def build_llm_decoy(self, fact: Fact) -> Fact:
        """Use LLM to generate a high-quality counterfactual decoy."""
        prompt = f"""Given this personal fact about a user:
"{fact.content}"

Generate ONE alternative version that:
1. Has the SAME structure and length
2. Replaces the key detail with a different but equally plausible value
3. Sounds natural and believable

Reply with ONLY the alternative fact, nothing else."""

        resp = await post_with_retry(
            self.client,
            f"{self.api_base}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.8,
                "max_tokens": 128,
            },
        )
        content = resp.json()["choices"][0]["message"]["content"].strip().strip('"')

        return Fact(
            content=content,
            topic=fact.topic + "_llm_decoy",
            key_value="[llm_generated]",
            category=fact.category,
            is_member=False,
        )

    async def build_decoy_pair(self, fact: Fact) -> DecoyPair:
        """Build a counterfactual decoy pair."""
        decoy = await self.build_llm_decoy(fact)
        return DecoyPair(fact=fact, decoy=decoy)

    async def close(self):
        await self.client.aclose()


# ═══════════════════════════════════════════════════════════════════════════
# Backward Compatibility Alias
# ═══════════════════════════════════════════════════════════════════════════

# Default to the new LLM-based builder
DecoyBuilder = LLMDecoyBuilder


# ═══════════════════════════════════════════════════════════════════════════
# CLI for Testing
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    """Test the LLMDecoyBuilder with sample inputs."""
    import argparse

    parser = argparse.ArgumentParser(description="Test DecoyBuilder pair generation with sample memories")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    args = parser.parse_args()

    # Pair-oriented test cases
    test_units = [
        MemoryUnit(
            content="Wang Xiaoming is interested in photography.",
            topic="hobby",
            key_value="photography",
            perlt_type=PerltType.DIALOGUE,
            user_id=0,
            source_key="test_0",
        ),
        MemoryUnit(
            content="Wang Xiaoming uses a smart office solution called iConnect.",
            topic="work_tool",
            key_value="iConnect",
            perlt_type=PerltType.DIALOGUE,
            user_id=0,
            source_key="test_1",
        ),
        MemoryUnit(
            content="Wang Xiaoming is currently a senior software engineer.",
            topic="job",
            key_value="senior software engineer",
            perlt_type=PerltType.DIALOGUE,
            user_id=0,
            source_key="test_2",
        ),
    ]

    print("=" * 70)
    print("Testing LLMDecoyBuilder pair generation")
    print("=" * 70)

    builder = LLMDecoyBuilder(temperature=args.temperature)

    try:
        for unit in test_units:
            print(f"\n📝 Original: {unit.content}")
            print(f"   topic={unit.topic} key_value={unit.key_value}")
            pair = await builder.build_decoy_pair_for_memory(unit)
            print(f"🔁 Pair topic: {pair.original.topic}")
            print(f"   original_key_value: {pair.original.key_value}")
            print(f"   decoy_key_value:    {pair.decoy.key_value}")
            print(f"   decoy_content:      {pair.decoy.content}")
            print("-" * 70)

    finally:
        await builder.close()

    print("\n Done!")


if __name__ == "__main__":
    asyncio.run(main())

"""Decoy builder — construct counterfactual facts for contrastive probing.

Two implementations:
- LegacyDecoyBuilder: Original implementation using hardcoded patterns (deprecated)
- LLMDecoyBuilder: New implementation using vLLM to generate high-quality counterparts
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional, Union

import httpx

from config import DEFAULT_API_BASE, DEFAULT_API_KEY, DEFAULT_MODEL
from http_utils import post_with_retry
from models import Fact, DecoyPair
from memory_unit import MemoryUnit, MemoryUnitPair, PerltType

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# LLM-based Decoy Builder (New)
# ═══════════════════════════════════════════════════════════════════════════

COUNTERFACTUAL_PROMPT = """\
You are an expert at creating counterfactual statements. Given a personal memory or fact, \
generate an alternative version that is DIFFERENT but equally plausible and natural.

## Requirements:
1. **Same structure and format**: Keep the same sentence structure, length, and style
2. **Different key information**: Change the core fact/value to something plausible but different
3. **Equally natural**: The alternative should sound just as natural and believable
4. **Same topic domain**: Stay in the same general category (if about food, stay with food; if about work, stay with work)
5. **No obvious contradictions**: Don't just negate (e.g., "doesn't like" instead of "likes")

## Examples:
Original: "The user's favorite color is blue."
Good: "The user's favorite color is green."
Bad: "The user doesn't have a favorite color." (negation)

Original: "I had lunch with my colleague Sarah at the Italian restaurant downtown yesterday."
Good: "I had lunch with my colleague Mike at the Thai restaurant near the office yesterday."
Bad: "I skipped lunch yesterday." (different structure)

Original: "The optimizer learning rate is 7e-4."
Good: "The optimizer learning rate is 3e-3."

Original: "My mother called me this morning to discuss my sister's wedding plans."
Good: "My father called me this afternoon to discuss my brother's graduation ceremony."

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
generate a decoy version and extract key metadata for structured comparison.

## Key Concept:
- **topic**: A SINGLE, atomic dimension/axis that can be varied (e.g., "hobby", "favorite color", "job title")
- **key_value**: The SINGLE value corresponding to that topic
- **decoy_content**: Change ONLY the key_value, keep ALL other details exactly the same

## Requirements:
1. **Single topic axis**: The topic must represent ONE specific dimension, not multiple combined concepts
2. **Prefer attributes over names**: Choose to vary the predicate/attribute (hobby, job, tool) rather than subject names when possible
3. **Minimal change**: The decoy changes ONLY the key_value; all other details remain identical
4. **Same structure**: Keep the exact same sentence structure, length, and style
5. **Equally natural**: The decoy should sound just as natural and believable
6. **No negation**: Don't just negate (e.g., "doesn't like" instead of "likes")

## Examples:

Original: "Wang Xiaoming is interested in photography."
Output:
{{
  "topic": "hobby",
  "original_key_value": "photography",
  "decoy_key_value": "painting",
  "decoy_content": "Wang Xiaoming is interested in painting."
}}
(Note: Change the hobby, keep the person name unchanged)

Original: "Wang Xiaoming uses a smart office solution called iConnect."
Output:
{{
  "topic": "office tool",
  "original_key_value": "iConnect",
  "decoy_key_value": "WorkFlow Pro",
  "decoy_content": "Wang Xiaoming uses a smart office solution called WorkFlow Pro."
}}
(Note: Change the tool name, keep the person name unchanged)

Original: "The user's favorite color is blue."
Output:
{{
  "topic": "favorite color",
  "original_key_value": "blue",
  "decoy_key_value": "green",
  "decoy_content": "The user's favorite color is green."
}}

Original: "The optimizer learning rate is 7e-4."
Output:
{{
  "topic": "learning rate",
  "original_key_value": "7e-4",
  "decoy_key_value": "3e-3",
  "decoy_content": "The optimizer learning rate is 3e-3."
}}

Bad examples (avoid these):
- Original: "Wang Xiaoming is interested in photography." → topic: "name", decoy: "Li Xiaohong is interested in photography." (should change hobby, not name)
- topic: "lunch meeting" with key_value: "Sarah, Italian restaurant" (multiple values combined - should be single axis)
- Changing multiple things at once

## Input:
{content}

## Output:
Return ONLY valid JSON in this exact shape, nothing else:
{{
  "topic": "single atomic topic/axis (prefer attribute over subject name)",
  "original_key_value": "the single value for that topic",
  "decoy_key_value": "different plausible value for the same topic",
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

    @staticmethod
    def _extract_json_text(content: str) -> str:
        text = content.strip()
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0]
        return text.strip()

    async def _call_llm_json(self, prompt: str, temperature: Optional[float] = None) -> dict:
        """Call vLLM and parse a JSON object response."""
        raw = await self._call_llm(prompt, temperature)
        return json.loads(self._extract_json_text(raw))

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

    async def build_hard_negative(
        self, content: str, temperature: Optional[float] = None
    ) -> str:
        """Generate a hard negative: same domain but different fact.

        Args:
            content: The original fact/memory content
            temperature: Optional override for generation temperature

        Returns:
            The hard negative content string
        """
        prompt = HARD_NEGATIVE_PROMPT.format(content=content)
        return await self._call_llm(prompt, temperature or 0.9)

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
        payload = await self._call_llm_json(prompt, temperature=self.temperature)

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
            raise ValueError("LLM did not return decoy_content for memory pair")
        if not decoy_key_value:
            raise ValueError("LLM did not return decoy_key_value for memory pair")
        if original_key_value and decoy_key_value.lower() == original_key_value.lower():
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

    async def build_hard_negative_for_fact(self, fact: Fact) -> Fact:
        """Build a hard negative for a Fact (legacy model)."""
        neg_content = await self.build_hard_negative(fact.content)

        return Fact(
            content=neg_content,
            topic=fact.topic + "_hard_neg",
            key_value="[llm_generated]",
            category=fact.category,
            is_member=False,
        )

    async def build_decoy_pair_for_fact(self, fact: Fact) -> DecoyPair:
        """Build a full decoy pair for a Fact."""
        decoy, hard_neg = await asyncio.gather(
            self.build_decoy_for_fact(fact),
            self.build_hard_negative_for_fact(fact),
        )
        return DecoyPair(fact=fact, decoy=decoy, hard_negative=hard_neg)

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
                        return (pair.fact, pair.decoy, pair.hard_negative)
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

    async def build_hard_negative(self, fact: Fact) -> Fact:
        """Generate a hard negative: same topic domain but entirely different fact."""
        prompt = f"""Given this fact about a user:
"{fact.content}"

Generate a DIFFERENT fact about the same general topic domain (e.g., if it's about food preference, generate a different food-related fact). The fact should:
1. Be about the same general domain
2. Contain DIFFERENT specific information
3. Be plausible as a real user fact

Reply with ONLY the fact, nothing else."""

        resp = await post_with_retry(
            self.client,
            f"{self.api_base}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.9,
                "max_tokens": 128,
            },
        )
        content = resp.json()["choices"][0]["message"]["content"].strip().strip('"')

        return Fact(
            content=content,
            topic=fact.topic + "_hard_neg",
            key_value="[llm_generated]",
            category=fact.category,
            is_member=False,
        )

    async def build_decoy_pair(self, fact: Fact) -> DecoyPair:
        """Build a full decoy pair with both counterfactual and hard negative."""
        decoy, hard_neg = await asyncio.gather(
            self.build_llm_decoy(fact),
            self.build_hard_negative(fact),
        )
        return DecoyPair(fact=fact, decoy=decoy, hard_negative=hard_neg)

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

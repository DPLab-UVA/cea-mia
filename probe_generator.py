"""Probe generator -- create multi-perspective probe families for CEA-MI.

Uses LLM to generate natural probing questions based on memory topic,
with separate prompts for each probe type. For a memory pair, both original
and decoy share the same question structure (only expected answers differ).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx

from config import DEFAULT_API_BASE, DEFAULT_API_KEY, DEFAULT_LLM_TIMEOUT, DEFAULT_MODEL
from http_utils import post_with_retry
from json_utils import JsonObjectError, loads_json_object
from models import Fact, Probe, ProbeType, DecoyPair
from memory_unit import MemoryUnit, MemoryUnitPair, PerltType

log = logging.getLogger(__name__)


class ShortDirectProbeSetError(JsonObjectError):
    """Raised when a direct probe set is valid but shorter than requested."""

    def __init__(self, expected: int, probe_specs: list[dict[str, str]]):
        self.expected = expected
        self.probe_specs = probe_specs
        super().__init__(
            f"expected at least {expected} probes, got {len(probe_specs)}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Probe Generation Prompts (one per probe type)
# ═══════════════════════════════════════════════════════════════════════════

DIRECT_RECALL_PROMPT = """\
Generate a pair of direct recall questions with IDENTICAL structure, one for each memory.

Memory A: "{content_a}" (topic: {topic}, value: {key_value_a})
Memory B: "{content_b}" (topic: {topic}, value: {key_value_b})

Requirements:
1. Both questions must have the EXACT same wording (since they ask about the same topic without mentioning values)
2. Directly ask about the topic without mentioning any specific value
3. Sound natural, like a user asking their personal assistant
4. If the memory contains contextual details (time, location, people, events) beyond the topic/value, \
incorporate them into the question to make it more specific. This helps distinguish similar memories.

Example 1 (simple):
Memory A: "Anna is interested in photography." (e.g., topic: hobby, value: photography)
Memory B: "Anna is interested in hiking." (topic: hobby, value: hiking)
Output: {{"question_a": "What's Anna's hobby?", "question_b": "What's Anna's hobby?"}}

Example 2 (with context - preferred when available):
Memory A: "John traveled to Arizona last summer for vacation." (topic: destination, value: Arizona)
Memory B: "John traveled to Colorado last summer for vacation." (topic: destination, value: Colorado)
Output: {{"question_a": "Where did John go for vacation last summer?", "question_b": "Where did John go for vacation last summer?"}}

Example 3 (with richer context):
Memory A: "During the 2023 conference, Dr. Wang presented her research on neural networks." (topic: research area, value: neural networks)
Memory B: "During the 2023 conference, Dr. Wang presented her research on computer vision." (topic: research area, value: computer vision)
Output: {{"question_a": "What research did Dr. Wang present at the 2023 conference?", "question_b": "What research did Dr. Wang present at the 2023 conference?"}}

Output ONLY valid JSON:
{{"question_a": "the question", "question_b": "the question"}}
"""

MULTI_DIRECT_RECALL_PROMPT = """\
Generate exactly {k} direct-recall probes for a memory membership inference attack.

Candidate memory statement:
"{content}"

Requirements:
1. Each probe must be a natural and specific direct recall question whose answer is a concise value entailed by the memory.
   The same question must also ask for the source/reason for the answer, using a short follow-up such as
   "How do you know that?", "Where did you get this information?", or "What memory tells you this?"
2. Do NOT ask yes/no questions.
3. Do NOT put the answer/key_value directly in the question.
4. Prefer probes that target different atomic topics or slots in the memory, such as person, location,
   date, event, relationship, organization, action, object, preference, or outcome.
5. Include non-answer context from the memory when it helps disambiguate the probe. Do not include the key_value itself as context.
6. If the memory contains fewer than {k} distinct atomic topics, first cover as many distinct
   topics as possible, then fill the remaining probes with natural paraphrases or different
   contextual framings of those available direct-recall question(s).
7. The source/reason follow-up should make it hard to answer from generic world knowledge alone; prefer asking
   what remembered fact, prior conversation, or stored information supports the answer.
8. Return exactly {k} probe objects.

For each probe:
- topic: the atomic slot being queried
- key_value: the concise expected answer if this memory is present
- question: the direct recall question plus a short source/reason follow-up

Example when k=5 and three distinct topics are available:
Memory: "Alice bought a blue backpack at Target."
Output:
{{
  "probes": [
    {{"topic": "person", "key_value": "Alice", "question": "Who bought a blue backpack at Target? How do you know that?"}},
    {{"topic": "item", "key_value": "blue backpack", "question": "What did Alice buy at Target? Where did you get this information?"}},
    {{"topic": "store", "key_value": "Target", "question": "Where did Alice buy the backpack? What memory tells you this?"}},
    {{"topic": "store", "key_value": "Target", "question": "Which store did Alice buy the backpack at? What prior information supports your answer?"}},
    {{"topic": "item", "key_value": "blue backpack", "question": "What kind of backpack did Alice buy at Target? How are you sure?"}}
  ]
}}

Output ONLY valid JSON:
{{
  "probes": [
    {{"topic": "atomic topic", "key_value": "expected answer", "question": "direct recall question plus source/reason follow-up"}}
  ]
}}
"""

INDIRECT_REASONING_PROMPT = """\
Generate a pair of indirect reasoning questions with IDENTICAL structure, one for each memory.

Memory A: "{content_a}" (topic: {topic}, value: {key_value_a})
Memory B: "{content_b}" (topic: {topic}, value: {key_value_b})

Requirements:
1. Both questions must have the EXACT same wording
2. Ask for help/advice/suggestion that naturally requires knowing the topic's value to answer
3. Do NOT directly ask about the topic itself
4. Be contextually relevant to the memories
5. If the memory contains contextual details (time, location, people, events) beyond the topic/value, \
reference them in the question to make it more specific and grounded in the original context.

Example 1 (simple):
Memory A: "Anna is interested in photography." (e.g., topic: hobby, value: photography)
Memory B: "Anna is interested in hiking." (topic: hobby, value: hiking)
Output: {{"question_a": "Anna has a free weekend, what activity would you suggest for her?", "question_b": "Anna has a free weekend, what activity would you suggest for her?"}}

Example 2 (with temporal context):
Memory A: "John traveled to Arizona last summer for vacation." (topic: destination, value: Arizona)
Memory B: "John traveled to Colorado last summer for vacation." (topic: destination, value: Colorado)
Output: {{"question_a": "John wants to revisit his favorite spots from last summer's trip. Can you suggest what he should pack?", "question_b": "John wants to revisit his favorite spots from last summer's trip. Can you suggest what he should pack?"}}

Example 3 (with event context):
Memory A: "During the team dinner, Lisa mentioned she prefers Italian cuisine." (topic: food preference, value: Italian cuisine)
Memory B: "During the team dinner, Lisa mentioned she prefers Japanese cuisine." (topic: food preference, value: Japanese cuisine)
Output: {{"question_a": "We're planning another team dinner and want to pick a restaurant Lisa would enjoy. Any suggestions?", "question_b": "We're planning another team dinner and want to pick a restaurant Lisa would enjoy. Any suggestions?"}}

Output ONLY valid JSON:
{{"question_a": "the question", "question_b": "the question"}}
"""

PROVENANCE_PROMPT = """\
Generate a pair of provenance questions with IDENTICAL structure, one for each memory.

Memory A: "{content_a}" (topic: {topic}, value: {key_value_a})
Memory B: "{content_b}" (topic: {topic}, value: {key_value_b})

Requirements:
1. Both questions must have the EXACT same wording
2. Ask about what the assistant knows regarding this topic
3. Optionally ask about the source of that information
4. Be contextually relevant to the memories
5. If the memory contains contextual details (e.g., time, location, people, events), \
include them in the question to ask about a specific piece of information rather than general knowledge.

Example 1 (simple):
Memory A: "Anna is interested in photography." (topic: hobby, value: photography)
Memory B: "Anna is interested in hiking." (topic: hobby, value: hiking)
Output: {{"question_a": "What do you know about Anna's hobbies, and how do you know that?", "question_b": "What do you know about Anna's hobbies, and how do you know that?"}}

Example 2 (with temporal context):
Memory A: "John traveled to Arizona last summer for vacation." (topic: destination, value: Arizona)
Memory B: "John traveled to Colorado last summer for vacation." (topic: destination, value: Colorado)
Output: {{"question_a": "What do you remember about John's vacation last summer? Where did he go?", "question_b": "What do you remember about John's vacation last summer? Where did he go?"}}

Example 3 (with event context):
Memory A: "During the project meeting, Tom proposed using Python for the backend." (topic: technology choice, value: Python)
Memory B: "During the project meeting, Tom proposed using Go for the backend." (topic: technology choice, value: Go)
Output: {{"question_a": "What do you know about Tom's proposal from the project meeting? What technology did he suggest?", "question_b": "What do you know about Tom's proposal from the project meeting? What technology did he suggest?"}}

Output ONLY valid JSON:
{{"question_a": "the question", "question_b": "the question"}}
"""

CONFIRMATION_PROMPT = """\
Generate a pair of confirmation questions with IDENTICAL structure, one for each memory.

Memory A: "{content_a}" (topic: {topic}, value: {key_value_a})
Memory B: "{content_b}" (topic: {topic}, value: {key_value_b})

Requirements:
1. Both questions must have the EXACT same structure, only differing in the value
2. Directly mention the specific value and ask for confirmation
3. Use phrases like "right?", "correct?", "isn't it?"
4. If the memory contains contextual details (e.g., time, location, people, events), \
include them in the question to confirm a specific fact rather than a general one.

Example 1 (simple):
Memory A: "Anna is interested in photography." (topic: hobby, value: photography)
Memory B: "Anna is interested in hiking." (topic: hobby, value: hiking)
Output: {{"question_a": "Anna's hobby is photography, right?", "question_b": "Anna's hobby is hiking, right?"}}

Example 2 (with temporal context):
Memory A: "John traveled to Arizona last summer for vacation." (topic: destination, value: Arizona)
Memory B: "John traveled to Colorado last summer for vacation." (topic: destination, value: Colorado)
Output: {{"question_a": "John went to Arizona for his vacation last summer, correct?", "question_b": "John went to Colorado for his vacation last summer, correct?"}}

Example 3 (with event context):
Memory A: "At the company retreat, Emily won the chess tournament." (topic: winner, value: chess tournament)
Memory B: "At the company retreat, Emily won the poker tournament." (topic: winner, value: poker tournament)
Output: {{"question_a": "Emily won the chess tournament at the company retreat, right?", "question_b": "Emily won the poker tournament at the company retreat, right?"}}

Output ONLY valid JSON:
{{"question_a": "question confirming key_value_a", "question_b": "question confirming key_value_b"}}
"""

class ProbeGenerator:
    """Generate probe questions using LLM for flexible, natural probing.

    For a memory pair (original, decoy), generates ONE set of questions
    that both share, except confirmation which needs value-specific questions.
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
            self._client = httpx.AsyncClient(timeout=DEFAULT_LLM_TIMEOUT)
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    # ── LLM Calls ─────────────────────────────────────────────────────────

    async def _call_llm(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: int = 256,
    ) -> str:
        """Call LLM API to generate text with retry on transient failures."""
        temp = temperature if temperature is not None else self.temperature
        resp = await post_with_retry(
            self.client,
            f"{self.api_base}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temp,
                "max_tokens": max_tokens,
            },
        )
        return resp.json()["choices"][0]["message"]["content"].strip().strip('"')

    # ── Individual Probe Generation Methods ───────────────────────────────

    def _parse_question_pair(self, raw: str) -> tuple[str, str]:
        """Parse JSON response containing question_a and question_b."""
        result = loads_json_object(raw, required_keys=("question_a", "question_b"))
        question_a = (result.get("question_a") or "").strip()
        question_b = (result.get("question_b") or "").strip()
        if not question_a or not question_b:
            raise JsonObjectError("question_a/question_b must be non-empty strings")
        return question_a, question_b

    def _parse_direct_recall_probe_set(self, raw: str, k: int) -> list[dict[str, str]]:
        """Parse JSON response containing a list of direct-recall probes."""
        result = loads_json_object(raw, required_keys=("probes",))
        probes = result.get("probes")
        if not isinstance(probes, list):
            raise JsonObjectError("probes must be a list")

        parsed = []
        for idx, item in enumerate(probes[:k], start=1):
            if not isinstance(item, dict):
                raise JsonObjectError(f"probe {idx} must be an object")
            topic = str(item.get("topic") or "").strip()
            key_value = str(item.get("key_value") or "").strip()
            question = str(item.get("question") or "").strip()
            if not topic or not key_value or not question:
                raise JsonObjectError(
                    f"probe {idx} must include non-empty topic, key_value, and question"
                )
            parsed.append({
                "topic": topic,
                "key_value": key_value,
                "question": question,
            })
        if len(parsed) < k:
            raise ShortDirectProbeSetError(k, parsed)
        return parsed

    @staticmethod
    def _build_direct_recall_probes(
        fact_id: str,
        probe_specs: list[dict[str, str]],
    ) -> list[Probe]:
        return [
            Probe(
                fact_id=fact_id,
                probe_type=ProbeType.DIRECT_RECALL,
                topic=spec["topic"],
                question=spec["question"],
                expected_if_member=spec["key_value"],
                perspective_idx=idx,
            )
            for idx, spec in enumerate(probe_specs)
        ]

    async def _generate_question_pair(self, prompt: str, probe_name: str) -> tuple[str, str]:
        attempts = max(1, self.max_retries + 1)
        last_error: Optional[Exception] = None

        for attempt in range(1, attempts + 1):
            raw = await self._call_llm(
                prompt,
                self.temperature if attempt == 1 else 0.0,
            )
            try:
                return self._parse_question_pair(raw)
            except JsonObjectError as exc:
                last_error = exc
                log.warning(
                    "Invalid probe JSON for %s (attempt %d/%d): %s | raw=%r",
                    probe_name,
                    attempt,
                    attempts,
                    exc,
                    raw[:500].replace("\n", "\\n"),
                )

        raise ValueError(f"Failed to generate valid {probe_name} probe JSON after {attempts} attempts") from last_error

    async def generate_direct_recall_probe_set(
        self,
        content: str,
        fact_id: str,
        k: int,
    ) -> list[Probe]:
        """Generate k direct-recall probes in one LLM call."""
        k = max(1, int(k))
        prompt = MULTI_DIRECT_RECALL_PROMPT.format(k=k, content=content)
        attempts = max(1, self.max_retries + 1)
        last_error: Optional[Exception] = None

        for attempt in range(1, attempts + 1):
            raw = await self._call_llm(
                prompt,
                self.temperature if attempt == 1 else 0.0,
                max_tokens=max(384, 160 * k),
            )
            try:
                probe_specs = self._parse_direct_recall_probe_set(raw, k)
                return self._build_direct_recall_probes(fact_id, probe_specs)
            except ShortDirectProbeSetError as exc:
                last_error = exc
                log.warning(
                    "Invalid multi direct probe JSON for fact %s (attempt %d/%d): %s | raw=%r",
                    fact_id,
                    attempt,
                    attempts,
                    exc,
                    raw[:500].replace("\n", "\\n"),
                )
            except JsonObjectError as exc:
                last_error = exc
                log.warning(
                    "Invalid multi direct probe JSON for fact %s (attempt %d/%d): %s | raw=%r",
                    fact_id,
                    attempt,
                    attempts,
                    exc,
                    raw[:500].replace("\n", "\\n"),
                )

        if (
            isinstance(last_error, ShortDirectProbeSetError)
            and last_error.probe_specs
        ):
            log.warning(
                "Using %d/%d generated direct probes for fact %s after %d attempts; "
                "fallback applies only because the JSON was valid but short",
                len(last_error.probe_specs),
                k,
                fact_id,
                attempts,
            )
            return self._build_direct_recall_probes(fact_id, last_error.probe_specs)

        raise ValueError(
            f"Failed to generate valid multi direct probe JSON after {attempts} attempts"
        ) from last_error

    async def generate_direct_recall(
        self,
        topic: str,
        content_a: str,
        key_value_a: str,
        content_b: str,
        key_value_b: str,
    ) -> tuple[str, str]:
        """Generate a pair of direct recall questions with identical structure."""
        prompt = DIRECT_RECALL_PROMPT.format(
            topic=topic,
            content_a=content_a,
            key_value_a=key_value_a,
            content_b=content_b,
            key_value_b=key_value_b,
        )
        return await self._generate_question_pair(prompt, "direct_recall")

    async def generate_indirect_reasoning(
        self,
        topic: str,
        content_a: str,
        key_value_a: str,
        content_b: str,
        key_value_b: str,
    ) -> tuple[str, str]:
        """Generate a pair of indirect reasoning questions with identical structure."""
        prompt = INDIRECT_REASONING_PROMPT.format(
            topic=topic,
            content_a=content_a,
            key_value_a=key_value_a,
            content_b=content_b,
            key_value_b=key_value_b,
        )
        return await self._generate_question_pair(prompt, "indirect_reasoning")

    async def generate_provenance(
        self,
        topic: str,
        content_a: str,
        key_value_a: str,
        content_b: str,
        key_value_b: str,
    ) -> tuple[str, str]:
        """Generate a pair of provenance questions with identical structure."""
        prompt = PROVENANCE_PROMPT.format(
            topic=topic,
            content_a=content_a,
            key_value_a=key_value_a,
            content_b=content_b,
            key_value_b=key_value_b,
        )
        return await self._generate_question_pair(prompt, "provenance")

    async def generate_confirmation(
        self,
        topic: str,
        content_a: str,
        key_value_a: str,
        content_b: str,
        key_value_b: str,
    ) -> tuple[str, str]:
        """Generate a pair of confirmation questions with identical structure."""
        prompt = CONFIRMATION_PROMPT.format(
            topic=topic,
            content_a=content_a,
            key_value_a=key_value_a,
            content_b=content_b,
            key_value_b=key_value_b,
        )
        return await self._generate_question_pair(prompt, "confirmation")

    # ── Helper Methods ────────────────────────────────────────────────────

    @staticmethod
    def _topic_root(topic: str | None) -> str:
        if not topic:
            return ""
        return str(topic).split("_")[0].strip()

    def _memory_unit_to_fact(self, unit: MemoryUnit) -> Fact:
        return Fact(
            id=unit.id,
            content=unit.content,
            topic=self._topic_root(unit.topic) or "",
            key_value=unit.key_value or "",
            category=unit.perlt_type.value,
            is_member=unit.is_member,
        )

    def _pair_to_facts(self, pair) -> tuple[Fact, Fact]:
        if isinstance(pair, DecoyPair):
            return pair.fact, pair.decoy
        if isinstance(pair, MemoryUnitPair):
            return self._memory_unit_to_fact(pair.original), self._memory_unit_to_fact(pair.decoy)
        if isinstance(pair, tuple) and len(pair) >= 2 and isinstance(pair[0], MemoryUnit):
            return self._memory_unit_to_fact(pair[0]), self._memory_unit_to_fact(pair[1])
        raise TypeError(f"Unsupported pair type for probe generation: {type(pair)!r}")

    # ── Main API ──────────────────────────────────────────────────────────

    async def generate_probe_family(self, pair) -> list[tuple[Probe, Probe]]:
        """Generate a complete probe family for a memory pair.

        For each probe type, generates ONE question based on the shared topic,
        then creates probe pairs where original and decoy share the same question
        but have different expected answers.

        Args:
            pair: A MemoryUnitPair, DecoyPair, or tuple of (original, decoy)

        Returns:
            List of (original_probe, decoy_probe) tuples for each probe type
        """
        fact, decoy = self._pair_to_facts(pair)
        topic = fact.topic

        # Generate all question pairs in parallel
        # Each method returns (question_a, question_b) with identical structure
        direct_pair, indirect_pair, provenance_pair, confirm_pair = (
            await asyncio.gather(
                self.generate_direct_recall(
                    topic, fact.content, fact.key_value, decoy.content, decoy.key_value
                ),
                self.generate_indirect_reasoning(
                    topic, fact.content, fact.key_value, decoy.content, decoy.key_value
                ),
                self.generate_provenance(
                    topic, fact.content, fact.key_value, decoy.content, decoy.key_value
                ),
                self.generate_confirmation(
                    topic, fact.content, fact.key_value, decoy.content, decoy.key_value
                ),
            )
        )
        fact_direct_q, decoy_direct_q = direct_pair
        fact_indirect_q, decoy_indirect_q = indirect_pair
        fact_provenance_q, decoy_provenance_q = provenance_pair
        fact_confirm_q, decoy_confirm_q = confirm_pair

        probe_pairs: list[tuple[Probe, Probe]] = []

        # 1. Direct recall (structurally identical questions)
        probe_pairs.append((
            Probe(
                fact_id=fact.id,
                probe_type=ProbeType.DIRECT_RECALL,
                topic=topic,
                question=fact_direct_q,
                expected_if_member=fact.key_value,
                perspective_idx=0,
            ),
            Probe(
                fact_id=fact.id,
                probe_type=ProbeType.DIRECT_RECALL,
                topic=topic,
                question=decoy_direct_q,
                expected_if_member=decoy.key_value,
                perspective_idx=0,
            ),
        ))

        # 2. Indirect reasoning (structurally identical questions)
        probe_pairs.append((
            Probe(
                fact_id=fact.id,
                probe_type=ProbeType.INDIRECT_REASONING,
                topic=topic,
                question=fact_indirect_q,
                expected_if_member=fact.key_value,
                perspective_idx=2,
            ),
            Probe(
                fact_id=fact.id,
                probe_type=ProbeType.INDIRECT_REASONING,
                topic=topic,
                question=decoy_indirect_q,
                expected_if_member=decoy.key_value,
                perspective_idx=2,
            ),
        ))

        # 3. Provenance (structurally identical questions)
        probe_pairs.append((
            Probe(
                fact_id=fact.id,
                probe_type=ProbeType.PROVENANCE,
                topic=topic,
                question=fact_provenance_q,
                expected_if_member=fact.key_value,
                perspective_idx=3,
            ),
            Probe(
                fact_id=fact.id,
                probe_type=ProbeType.PROVENANCE,
                topic=topic,
                question=decoy_provenance_q,
                expected_if_member=decoy.key_value,
                perspective_idx=3,
            ),
        ))

        # 4. Confirmation (different questions for fact and decoy, each mentions its value)
        probe_pairs.append((
            Probe(
                fact_id=fact.id,
                probe_type=ProbeType.CONFIRMATION,
                topic=topic,
                question=fact_confirm_q,
                expected_if_member=fact.key_value,
                perspective_idx=4,
            ),
            Probe(
                fact_id=fact.id,
                probe_type=ProbeType.CONFIRMATION,
                topic=topic,
                question=decoy_confirm_q,
                expected_if_member=decoy.key_value,
                perspective_idx=4,
            ),
        ))

        return probe_pairs

    async def generate_probe_family_for_memory_pair(
        self, pair: MemoryUnitPair | tuple[MemoryUnit, MemoryUnit]
    ) -> list[tuple[Probe, Probe]]:
        """Generate a probe family from a pair of MemoryUnit objects."""
        return await self.generate_probe_family(pair)


# ═══════════════════════════════════════════════════════════════════════════
# CLI for Testing
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    """Test the ProbeGenerator with sample MemoryUnitPairs."""
    import argparse

    parser = argparse.ArgumentParser(description="Test ProbeGenerator with sample memory pairs")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    args = parser.parse_args()

    # Test cases based on decoy_builder output
    test_pairs = [
        MemoryUnitPair(
            original=MemoryUnit(
                id="test_0",
                content="Wang Xiaoming is interested in photography.",
                topic="hobby",
                key_value="photography",
                perlt_type=PerltType.DIALOGUE,
                user_id=0,
                source_key="test_0",
                is_member=True,
            ),
            decoy=MemoryUnit(
                id="test_0_decoy",
                content="Wang Xiaoming is interested in hiking.",
                topic="hobby",
                key_value="hiking",
                perlt_type=PerltType.DIALOGUE,
                user_id=0,
                source_key="test_0",
                is_member=False,
            ),
        ),
        MemoryUnitPair(
            original=MemoryUnit(
                id="test_1",
                content="Wang Xiaoming uses a smart office solution called iConnect.",
                topic="office tool",
                key_value="iConnect",
                perlt_type=PerltType.DIALOGUE,
                user_id=0,
                source_key="test_1",
                is_member=True,
            ),
            decoy=MemoryUnit(
                id="test_1_decoy",
                content="Wang Xiaoming uses a smart office solution called OfficePro Suite.",
                topic="office tool",
                key_value="OfficePro Suite",
                perlt_type=PerltType.DIALOGUE,
                user_id=0,
                source_key="test_1",
                is_member=False,
            ),
        ),
        MemoryUnitPair(
            original=MemoryUnit(
                id="test_2",
                content="Wang Xiaoming is currently a senior software engineer.",
                topic="job title",
                key_value="senior software engineer",
                perlt_type=PerltType.DIALOGUE,
                user_id=0,
                source_key="test_2",
                is_member=True,
            ),
            decoy=MemoryUnit(
                id="test_2_decoy",
                content="Wang Xiaoming is currently a principal software engineer.",
                topic="job title",
                key_value="principal software engineer",
                perlt_type=PerltType.DIALOGUE,
                user_id=0,
                source_key="test_2",
                is_member=False,
            ),
        ),
    ]

    print("=" * 70)
    print("Testing ProbeGenerator with LLM-generated probes")
    print("=" * 70)

    generator = ProbeGenerator(temperature=args.temperature)

    try:
        for pair in test_pairs:
            print(f"\n{'='*70}")
            print(f"📝 Original: {pair.original.content}")
            print(f"   topic={pair.original.topic} key_value={pair.original.key_value}")
            print(f"🔁 Decoy: {pair.decoy.content}")
            print(f"   decoy_key_value={pair.decoy.key_value}")
            print("-" * 70)

            probe_pairs = await generator.generate_probe_family(pair)

            for fact_probe, decoy_probe in probe_pairs:
                print(f"\n[{fact_probe.probe_type.value}]")
                if fact_probe.question == decoy_probe.question:
                    print(f"  Q (shared): {fact_probe.question}")
                else:
                    print(f"  Q (original): {fact_probe.question}")
                    print(f"  Q (decoy):    {decoy_probe.question}")
                print(f"  Expected (original): {fact_probe.expected_if_member}")
                print(f"  Expected (decoy):    {decoy_probe.expected_if_member}")

            print("-" * 70)

    finally:
        await generator.close()

    print("\n Done!")


if __name__ == "__main__":
    asyncio.run(main())

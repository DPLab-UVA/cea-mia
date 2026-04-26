"""Probe generator -- create multi-perspective probe families for CEA-MI.

Uses LLM to generate natural probing questions based on memory topic,
with separate prompts for each probe type. For a memory pair, both original
and decoy share the same question structure (only expected answers differ).
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

import httpx

from config import DEFAULT_API_BASE, DEFAULT_API_KEY, DEFAULT_MODEL
from http_utils import post_with_retry
from models import Fact, Probe, ProbeType, DecoyPair
from memory_unit import MemoryUnit, MemoryUnitPair, PerltType

log = logging.getLogger(__name__)


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

PARAPHRASE_PROMPT = """\
Rephrase this question {n} different ways. Keep the same meaning and intent.
The question is asking about personal information, so keep it natural and conversational.

Original: {question}

Output ONLY the rephrased questions, one per line, numbered 1-{n}. No explanations.
"""


class ProbeGenerator:
    """Generate probe questions using LLM for flexible, natural probing.

    For a memory pair (original, decoy), generates ONE set of questions
    that both share (except confirmation which needs value-specific questions).
    """

    def __init__(
        self,
        api_base: str = DEFAULT_API_BASE,
        api_key: str = DEFAULT_API_KEY,
        model: str = DEFAULT_MODEL,
        num_paraphrases: int = 3,
        temperature: float = 0.7,
    ):
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.num_paraphrases = num_paraphrases
        self.temperature = temperature
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

    # ── LLM Calls ─────────────────────────────────────────────────────────

    async def _call_llm(self, prompt: str, temperature: Optional[float] = None) -> str:
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
                "max_tokens": 256,
            },
        )
        return resp.json()["choices"][0]["message"]["content"].strip().strip('"')

    # ── Individual Probe Generation Methods ───────────────────────────────

    def _parse_question_pair(self, raw: str) -> tuple[str, str]:
        """Parse JSON response containing question_a and question_b."""
        text = raw.strip()
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0]
        result = json.loads(text.strip())
        return result["question_a"], result["question_b"]

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
        raw = await self._call_llm(prompt)
        return self._parse_question_pair(raw)

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
        raw = await self._call_llm(prompt)
        return self._parse_question_pair(raw)

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
        raw = await self._call_llm(prompt)
        return self._parse_question_pair(raw)

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
        raw = await self._call_llm(prompt)
        return self._parse_question_pair(raw)

    async def generate_paraphrases(self, base_question: str, n: int = 3) -> list[str]:
        """Generate paraphrases of a question."""
        if n <= 0:
            return []
        prompt = PARAPHRASE_PROMPT.format(question=base_question, n=n)
        try:
            text = await self._call_llm(prompt, temperature=0.9)
            lines = [
                line.strip().lstrip("0123456789.)- ")
                for line in text.strip().split("\n")
                if line.strip()
            ]
            return [line for line in lines if len(line) > 10][:n]
        except Exception as e:
            log.warning("Failed to generate paraphrases: %s", e)
            return [base_question]

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
                question=fact_direct_q,
                expected_if_member=fact.key_value,
                perspective_idx=0,
                paraphrase_idx=0,
            ),
            Probe(
                fact_id=fact.id,
                probe_type=ProbeType.DIRECT_RECALL,
                question=decoy_direct_q,
                expected_if_member=decoy.key_value,
                perspective_idx=0,
                paraphrase_idx=0,
            ),
        ))

        # 2. Paraphrase variants of direct recall
        if self.num_paraphrases > 0:
            paraphrases = await self.generate_paraphrases(fact_direct_q, self.num_paraphrases)
            for i, pq in enumerate(paraphrases):
                probe_pairs.append((
                    Probe(
                        fact_id=fact.id,
                        probe_type=ProbeType.PARAPHRASE,
                        question=pq,
                        expected_if_member=fact.key_value,
                        perspective_idx=1,
                        paraphrase_idx=i,
                    ),
                    Probe(
                        fact_id=fact.id,
                        probe_type=ProbeType.PARAPHRASE,
                        question=pq,
                        expected_if_member=decoy.key_value,
                        perspective_idx=1,
                        paraphrase_idx=i,
                    ),
                ))

        # 3. Indirect reasoning (structurally identical questions)
        probe_pairs.append((
            Probe(
                fact_id=fact.id,
                probe_type=ProbeType.INDIRECT_REASONING,
                question=fact_indirect_q,
                expected_if_member=fact.key_value,
                perspective_idx=2,
                paraphrase_idx=0,
            ),
            Probe(
                fact_id=fact.id,
                probe_type=ProbeType.INDIRECT_REASONING,
                question=decoy_indirect_q,
                expected_if_member=decoy.key_value,
                perspective_idx=2,
                paraphrase_idx=0,
            ),
        ))

        # 4. Provenance (structurally identical questions)
        probe_pairs.append((
            Probe(
                fact_id=fact.id,
                probe_type=ProbeType.PROVENANCE,
                question=fact_provenance_q,
                expected_if_member=fact.key_value,
                perspective_idx=3,
                paraphrase_idx=0,
            ),
            Probe(
                fact_id=fact.id,
                probe_type=ProbeType.PROVENANCE,
                question=decoy_provenance_q,
                expected_if_member=decoy.key_value,
                perspective_idx=3,
                paraphrase_idx=0,
            ),
        ))

        # 5. Confirmation (different questions for fact and decoy, each mentions its value)
        probe_pairs.append((
            Probe(
                fact_id=fact.id,
                probe_type=ProbeType.CONFIRMATION,
                question=fact_confirm_q,
                expected_if_member=fact.key_value,
                perspective_idx=4,
                paraphrase_idx=0,
            ),
            Probe(
                fact_id=fact.id,
                probe_type=ProbeType.CONFIRMATION,
                question=decoy_confirm_q,
                expected_if_member=decoy.key_value,
                perspective_idx=4,
                paraphrase_idx=0,
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

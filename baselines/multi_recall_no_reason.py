"""MRMMIA-style direct recall baseline without reason/source follow-up probes."""
from __future__ import annotations

import logging
from typing import Optional

from config import DEFAULT_API_BASE, DEFAULT_API_KEY, DEFAULT_MODEL
from feature_extractor import FeatureExtractor
from json_utils import JsonObjectError
from models import Probe, ProbeResult, ProbeType
from multi_probe_attack import MultiProbeDirectAttack
from probe_generator import ProbeGenerator, ShortDirectProbeSetError

log = logging.getLogger(__name__)


MULTI_RECALL_NO_REASON_PROMPT = """\
Generate exactly {k} direct-recall probes for a memory membership inference attack.

Candidate memory statement:
"{content}"

Requirements:
1. Each probe must be a natural and specific direct recall question whose answer is a concise value entailed by the memory.
2. Do NOT ask yes/no questions.
3. Do NOT ask for the source, evidence, reason, memory, or provenance of the answer.
4. Do NOT include follow-up questions such as "How do you know that?", "Where did you get this information?",
   "What memory tells you this?", or similar.
5. Do NOT put the answer/key_value directly in the question.
6. Prefer probes that target different atomic topics or slots in the memory, such as person, location,
   date, event, relationship, organization, action, object, preference, or outcome.
7. Include non-answer context from the memory when it helps disambiguate the probe. Do not include the key_value itself as context.
8. If the memory contains fewer than {k} distinct atomic topics, first cover as many distinct
   topics as possible, then fill the remaining probes with natural paraphrases or different
   contextual framings of those available direct-recall question(s).
9. Return exactly {k} probe objects.

For each probe:
- topic: the atomic slot being queried
- key_value: the concise expected answer if this memory is present
- question: one direct recall question only, with no source/reason/provenance follow-up

Example when k=5 and three distinct topics are available:
Memory: "Alice bought a blue backpack at Target."
Output:
{{
  "probes": [
    {{"topic": "person", "key_value": "Alice", "question": "Who bought a blue backpack at Target?"}},
    {{"topic": "item", "key_value": "blue backpack", "question": "What did Alice buy at Target?"}},
    {{"topic": "store", "key_value": "Target", "question": "Where did Alice buy the backpack?"}},
    {{"topic": "store", "key_value": "Target", "question": "Which store did Alice buy the backpack at?"}},
    {{"topic": "item", "key_value": "blue backpack", "question": "What kind of backpack did Alice buy at Target?"}}
  ]
}}

Output ONLY valid JSON:
{{
  "probes": [
    {{"topic": "atomic topic", "key_value": "expected answer", "question": "direct recall question"}}
  ]
}}
"""


class NoReasonRecallProbeGenerator(ProbeGenerator):
    """Generate old-style direct recall probes without reason/source follow-ups."""

    async def generate_direct_recall_probe_set(
        self,
        content: str,
        fact_id: str,
        k: int,
    ) -> list[Probe]:
        k = max(1, int(k))
        prompt = MULTI_RECALL_NO_REASON_PROMPT.format(k=k, content=content)
        attempts = max(1, self.max_retries + 1)
        last_error: Optional[Exception] = None

        for attempt in range(1, attempts + 1):
            raw = await self._call_llm(
                prompt,
                self.temperature if attempt == 1 else 0.0,
                max_tokens=max(384, 128 * k),
            )
            try:
                probe_specs = self._parse_direct_recall_probe_set(raw, k)
                return self._build_direct_recall_probes(fact_id, probe_specs)
            except ShortDirectProbeSetError as exc:
                last_error = exc
                log.warning(
                    "Invalid no-reason recall probe JSON for fact %s (attempt %d/%d): %s | raw=%r",
                    fact_id,
                    attempt,
                    attempts,
                    exc,
                    raw[:500].replace("\n", "\\n"),
                )
            except JsonObjectError as exc:
                last_error = exc
                log.warning(
                    "Invalid no-reason recall probe JSON for fact %s (attempt %d/%d): %s | raw=%r",
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
                "Using %d/%d generated no-reason recall probes for fact %s after %d attempts; "
                "fallback applies only because the JSON was valid but short",
                len(last_error.probe_specs),
                k,
                fact_id,
                attempts,
            )
            return self._build_direct_recall_probes(fact_id, last_error.probe_specs)

        raise ValueError(
            f"Failed to generate valid no-reason recall probe JSON after {attempts} attempts"
        ) from last_error


class MultiRecallNoReasonBaseline(MultiProbeDirectAttack):
    """MRMMIA-style attack using old direct recall probes without reason follow-ups."""

    name = "multi_recall_no_reason"

    def __init__(
        self,
        api_base: str = DEFAULT_API_BASE,
        api_key: str = DEFAULT_API_KEY,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        probe_concurrency: int = 40,
        response_scorer: str = "rules",
        direct_probe_k: int = 5,
    ):
        super().__init__(
            api_base=api_base,
            api_key=api_key,
            model=model,
            temperature=temperature,
            probe_concurrency=probe_concurrency,
            response_scorer=response_scorer,
            direct_probe_k=direct_probe_k,
            memory_statement_judge=True,
        )
        self.probe_gen = NoReasonRecallProbeGenerator(
            api_base=api_base,
            api_key=api_key,
            model=model,
            temperature=temperature,
        )

    async def _score_response_for_candidate(
        self,
        result: ProbeResult,
        candidate_statement: str,
        topic: str,
        key_value: str,
        probe_type: ProbeType,
    ) -> float:
        if self.response_scorer == "rules":
            return FeatureExtractor.score_response(
                result.response,
                key_value,
                probe_type,
            )

        if self.response_scorer != "llm" or self.llm_judge is None:
            raise ValueError(f"Unknown response_scorer: {self.response_scorer}")

        return await self.llm_judge.judge_recall_no_reason(
            candidate_statement=candidate_statement,
            topic=topic,
            key_value=key_value,
            question=result.probe.question,
            response=result.response,
        )

    async def _extract_no_contrastive_features(
        self,
        candidate_statement: str,
        topic: str,
        key_value: str,
        results: list[ProbeResult],
        probe_type: ProbeType,
    ) -> dict:
        features = await super()._extract_no_contrastive_features(
            candidate_statement,
            topic,
            key_value,
            results,
            probe_type,
        )
        features["probe_style"] = "direct_recall_no_reason"
        if self.response_scorer == "llm":
            features["response_scorer"] = "llm_key_value_no_reason"
        return features

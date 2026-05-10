"""Direct multi-probe membership attack without contrastive decoys."""
from __future__ import annotations

import asyncio
import logging
import math
from typing import Optional

from config import DEFAULT_API_BASE, DEFAULT_API_KEY, DEFAULT_MODEL
from feature_extractor import FeatureExtractor
from llm_response_judge import LLMResponseJudge
from memory_attack_utils import failed_prediction_for_unit
from memory_unit import MemoryUnit
from models import MembershipPrediction, Probe, ProbeResult, ProbeType, RoundEvidence
from probe_generator import ProbeGenerator

log = logging.getLogger(__name__)


async def query_agent(agent, message: str, access_level: str, target: str) -> dict:
    """Unified query interface across supported memory targets."""
    result = await agent.query(message, access_level=access_level)

    if access_level == "whitebox" and "recalled_memories" not in result and hasattr(agent, "recall"):
        recalled = agent.recall(message)
        result["recalled_memories"] = [
            {
                "type": "retrieved",
                "content": item.get("content", ""),
                "similarity": item.get("similarity", 0.0),
                "metadata": item.get("metadata", {}),
            }
            for item in recalled
            if isinstance(item, dict)
        ]

    return result


class MultiProbeDirectAttack:
    """Generate k direct-recall probes in one LLM call and average their scores."""

    name = "multi_probe"

    def __init__(
        self,
        api_base: str = DEFAULT_API_BASE,
        api_key: str = DEFAULT_API_KEY,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        probe_concurrency: int = 40,
        target_query_concurrency: Optional[int] = None,
        response_scorer: str = "rules",
        direct_probe_k: int = 5,
        memory_statement_judge: bool = False,
    ):
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.probe_concurrency = max(1, probe_concurrency)
        self.target_query_concurrency = (
            max(1, int(target_query_concurrency))
            if target_query_concurrency is not None
            else None
        )
        self.target_query_semaphore = (
            asyncio.Semaphore(self.target_query_concurrency)
            if self.target_query_concurrency
            else None
        )
        self.response_scorer = response_scorer
        self.direct_probe_k = max(1, int(direct_probe_k))
        self.memory_statement_judge = memory_statement_judge
        self.probe_gen = ProbeGenerator(
            api_base=api_base,
            api_key=api_key,
            model=model,
            temperature=temperature,
        )
        self.feat_ext = FeatureExtractor()
        self.llm_judge = (
            LLMResponseJudge(
                api_base=api_base,
                api_key=api_key,
                model=model,
                temperature=0.0,
            )
            if response_scorer == "llm"
            else None
        )

    async def close(self):
        await self.probe_gen.close()
        if self.llm_judge:
            await self.llm_judge.close()

    @staticmethod
    def _logprob_confidence(mean_logprob: float) -> float:
        try:
            return 1.0 / (1.0 + math.exp(-(mean_logprob + 2.0)))
        except OverflowError:
            return 0.0 if mean_logprob < 0 else 1.0

    async def _execute_probe(
        self,
        agent,
        probe: Probe,
        access_level: str,
        target: str,
    ) -> ProbeResult:
        if self.target_query_semaphore is None:
            resp = await query_agent(agent, probe.question, access_level, target)
        else:
            async with self.target_query_semaphore:
                resp = await query_agent(agent, probe.question, access_level, target)
        memory_metadata = None
        if access_level == "whitebox":
            memory_metadata = {
                "recalled_memories": resp.get("recalled_memories", []),
                "recall_scores": resp.get("recall_scores", []),
                "memory_stats": resp.get("memory_stats"),
            }

        return ProbeResult(
            probe=probe,
            response=resp.get("response", ""),
            latency_ms=resp.get("latency_ms", 0),
            logprobs=resp.get("logprobs"),
            mean_logprob=resp.get("mean_logprob"),
            recall_triggered=resp.get("recall_triggered") if access_level == "whitebox" else None,
            recall_top_similarity=resp.get("recall_top_similarity") if access_level == "whitebox" else None,
            recall_hit_count=resp.get("recall_hit_count") if access_level == "whitebox" else None,
            memory_metadata=memory_metadata,
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

        return await self.llm_judge.judge(
            candidate_statement=candidate_statement,
            topic=topic,
            key_value=key_value,
            question=result.probe.question,
            response=result.response,
            probe_type=probe_type,
        )

    async def _score_recalled_memories_for_candidate(
        self,
        result: ProbeResult,
        candidate_statement: str,
        topic: str,
        key_value: str,
    ) -> float:
        if self.response_scorer == "rules":
            return self.feat_ext.score_recalled_memories(result, key_value)

        if self.response_scorer != "llm" or self.llm_judge is None:
            raise ValueError(f"Unknown response_scorer: {self.response_scorer}")

        contents = self.feat_ext.recalled_memory_contents(result)
        if not contents:
            return 0.0

        if self.memory_statement_judge:
            scores = await asyncio.gather(*[
                self.llm_judge.judge_memory_statement(
                    candidate_statement=candidate_statement,
                    question=result.probe.question,
                    memory_content=content,
                )
                for content in contents
            ])
            return max(scores) if scores else 0.0

        scores = await asyncio.gather(*[
            self.llm_judge.judge_memory(
                candidate_statement=candidate_statement,
                topic=topic,
                key_value=key_value,
                question=result.probe.question,
                memory_content=content,
            )
            for content in contents
        ])
        return max(scores) if scores else 0.0

    async def _extract_no_contrastive_features(
        self,
        candidate_statement: str,
        topic: str,
        key_value: str,
        results: list[ProbeResult],
        probe_type: ProbeType,
    ) -> dict:
        response_scores = await asyncio.gather(*[
            self._score_response_for_candidate(
                result,
                candidate_statement,
                topic,
                key_value,
                probe_type,
            )
            for result in results
        ])
        response_mean = sum(response_scores) / len(response_scores) if response_scores else 0.0
        features = {
            "response_scorer": self.response_scorer,
            "fact_response_scores": list(response_scores),
            "fact_response_score_mean": response_mean,
            "delta_response_score": response_mean,
            "fact_similarity_mean": response_mean,
            "decoy_similarity_mean": 0.0,
            "delta_similarity": response_mean,
            "no_contrastive": True,
        }

        flp = [r.mean_logprob for r in results if r.mean_logprob is not None]
        if flp:
            features["delta_logprob"] = self._logprob_confidence(sum(flp) / len(flp))

        if any(r.memory_metadata is not None for r in results):
            memory_scores = await asyncio.gather(*[
                self._score_recalled_memories_for_candidate(
                    result,
                    candidate_statement,
                    topic,
                    key_value,
                )
                for result in results
            ])
            self.feat_ext.add_memory_statement_features(
                features,
                list(memory_scores),
                [],
                self.response_scorer,
            )
            features["fact_memory_candidate_statement"] = candidate_statement
            features["fact_memory_key_value"] = key_value

        return features

    async def _attack_unit(
        self,
        agent,
        unit: MemoryUnit,
        access_level: str,
        target: str,
    ) -> MembershipPrediction:
        try:
            probes = await self.probe_gen.generate_direct_recall_probe_set(
                content=unit.content,
                fact_id=unit.id,
                k=self.direct_probe_k,
            )
        except Exception as exc:
            log.warning(
                "Counting unit %s as score=0: multi direct probe generation failed: %s",
                unit.id,
                exc,
            )
            return failed_prediction_for_unit(unit, "probe_generation", exc)

        evidence_trail: list[RoundEvidence] = []
        for round_idx, probe in enumerate(probes):
            result = await self._execute_probe(agent, probe, access_level, target)
            topic = probe.topic or ""
            key_value = probe.expected_if_member or ""
            features = await self._extract_no_contrastive_features(
                unit.content,
                topic,
                key_value,
                [result],
                ProbeType.DIRECT_RECALL,
            )
            features["probe_topic"] = topic
            features["probe_key_value"] = key_value
            features["direct_probe_k"] = self.direct_probe_k
            score = self.feat_ext.compute_round_score(features)
            evidence_trail.append(
                RoundEvidence(
                    fact_id=unit.id,
                    round_idx=round_idx,
                    probe_type=ProbeType.DIRECT_RECALL,
                    score_fact=features.get("fact_similarity_mean", 0.0),
                    score_decoy=0.0,
                    delta_score=score,
                    features=features,
                    fact_results=[result],
                )
            )

        score = (
            sum(e.delta_score for e in evidence_trail) / len(evidence_trail)
            if evidence_trail
            else 0.0
        )
        return MembershipPrediction(
            fact_id=unit.id,
            is_member_true=unit.is_member,
            score=score,
            is_member_pred=score > 0.5,
            evidence_trail=evidence_trail,
            num_rounds_used=len(evidence_trail),
        )

    async def attack(self, agent, units, access_level: str, target: str) -> list[MembershipPrediction]:
        semaphore = asyncio.Semaphore(self.probe_concurrency)

        async def attack_one(idx: int, unit: MemoryUnit) -> MembershipPrediction:
            async with semaphore:
                try:
                    pred = await self._attack_unit(agent, unit, access_level, target)
                except Exception as exc:
                    log.error("[MultiProbeDirect] unit %s failed: %s", unit.id, exc, exc_info=True)
                    pred = failed_prediction_for_unit(unit, "attack", exc)

                if idx % 10 == 0:
                    log.info("[MultiProbeDirect] %d/%d done", idx, len(units))
                return pred

        tasks = [attack_one(i, unit) for i, unit in enumerate(units, start=1)]
        return list(await asyncio.gather(*tasks))

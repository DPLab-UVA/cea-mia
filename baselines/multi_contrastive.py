"""Legacy contrastive natural attack exposed as a baseline."""
from __future__ import annotations

import asyncio
import logging

from baselines.decoy_builder import LLMDecoyBuilder
from baselines.probe_batches import group_probe_pairs_by_round
from config import DEFAULT_API_BASE, DEFAULT_API_KEY, DEFAULT_MODEL
from feature_extractor import FeatureExtractor
from llm_response_judge import LLMResponseJudge
from memory_attack_utils import failed_prediction_for_unit, memory_to_fact
from memory_unit import MemoryUnit
from models import MembershipPrediction, ProbeResult, RoundEvidence
from multi_probe_attack import query_agent
from probe_generator import ProbeGenerator

log = logging.getLogger(__name__)


class MultiContrastiveBaseline:
    """The original natural attack: build a decoy and probe fact vs. decoy."""

    name = "multi_contrastive"

    def __init__(
        self,
        api_base: str = DEFAULT_API_BASE,
        api_key: str = DEFAULT_API_KEY,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        probe_concurrency: int = 40,
        response_scorer: str = "rules",
    ):
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.probe_concurrency = max(1, probe_concurrency)
        self.response_scorer = response_scorer
        self.decoy_builder = LLMDecoyBuilder(
            api_base=api_base,
            api_key=api_key,
            model=model,
            temperature=temperature,
        )
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
        await self.decoy_builder.close()
        await self.probe_gen.close()
        if self.llm_judge:
            await self.llm_judge.close()

    async def _execute_probe(self, agent, probe, access_level: str, target: str) -> ProbeResult:
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

    async def _add_whitebox_memory_statement_features(
        self,
        features: dict,
        pair,
        fact_results: list[ProbeResult],
        decoy_results: list[ProbeResult],
    ) -> dict:
        if not any(r.memory_metadata is not None for r in fact_results + decoy_results):
            return features

        fact_statement = pair.original.content
        decoy_statement = pair.decoy.content
        fact_topic = pair.original.topic or ""
        decoy_topic = pair.decoy.topic or fact_topic
        fact_key_value = pair.original.key_value or ""
        decoy_key_value = pair.decoy.key_value or ""

        recalled_results = fact_results + decoy_results
        fact_scores = await asyncio.gather(*[
            self._score_recalled_memories_for_candidate(
                result,
                fact_statement,
                fact_topic,
                fact_key_value,
            )
            for result in recalled_results
        ])
        decoy_scores = await asyncio.gather(*[
            self._score_recalled_memories_for_candidate(
                result,
                decoy_statement,
                decoy_topic,
                decoy_key_value,
            )
            for result in recalled_results
        ])

        self.feat_ext.add_memory_statement_features(
            features,
            list(fact_scores),
            list(decoy_scores),
            self.response_scorer,
        )
        features["fact_memory_candidate_statement"] = fact_statement
        features["decoy_memory_candidate_statement"] = decoy_statement
        features["fact_memory_key_value"] = fact_key_value
        features["decoy_memory_key_value"] = decoy_key_value
        return features

    async def _extract_round_features(
        self,
        pair,
        fact_results: list[ProbeResult],
        decoy_results: list[ProbeResult],
        probe_type,
    ) -> dict:
        fact = memory_to_fact(pair.original)
        if self.response_scorer == "rules":
            features = self.feat_ext.extract_round_features(
                fact,
                fact_results,
                decoy_results,
                probe_type,
            )
            features["response_scorer"] = "rules"
            return await self._add_whitebox_memory_statement_features(
                features,
                pair,
                fact_results,
                decoy_results,
            )

        if self.response_scorer != "llm" or self.llm_judge is None:
            raise ValueError(f"Unknown response_scorer: {self.response_scorer}")

        fact_statement = pair.original.content
        decoy_statement = pair.decoy.content
        fact_topic = pair.original.topic or fact.topic or ""
        decoy_topic = pair.decoy.topic or fact_topic
        fact_key_value = pair.original.key_value or fact.key_value or ""
        decoy_key_value = pair.decoy.key_value or ""

        fact_judgments = await asyncio.gather(*[
            self.llm_judge.judge(
                candidate_statement=fact_statement,
                topic=fact_topic,
                key_value=fact_key_value,
                question=result.probe.question,
                response=result.response,
                probe_type=probe_type,
            )
            for result in fact_results
        ])
        decoy_judgments = await asyncio.gather(*[
            self.llm_judge.judge(
                candidate_statement=decoy_statement,
                topic=decoy_topic,
                key_value=decoy_key_value,
                question=result.probe.question,
                response=result.response,
                probe_type=probe_type,
            )
            for result in decoy_results
        ])

        features = self.feat_ext.extract_round_features_from_scores(
            list(fact_judgments),
            list(decoy_judgments),
            fact_results,
            decoy_results,
        )
        features["response_scorer"] = "llm"
        features["fact_candidate_statement"] = fact_statement
        features["decoy_candidate_statement"] = decoy_statement
        features["fact_topic"] = fact_topic
        features["decoy_topic"] = decoy_topic
        features["fact_key_value"] = fact_key_value
        features["decoy_key_value"] = decoy_key_value
        return await self._add_whitebox_memory_statement_features(
            features,
            pair,
            fact_results,
            decoy_results,
        )

    async def _attack_unit(
        self,
        agent,
        unit: MemoryUnit,
        access_level: str,
        target: str,
    ) -> MembershipPrediction:
        try:
            pair = await self.decoy_builder.build_decoy_pair_for_memory(unit)
        except Exception as exc:
            log.warning("Counting unit %s as score=0: decoy generation failed: %s", unit.id, exc)
            return failed_prediction_for_unit(unit, "decoy_generation", exc)

        if not pair.original.key_value or not pair.decoy.key_value:
            return failed_prediction_for_unit(
                unit,
                "decoy_generation",
                "missing key_value in generated pair",
            )

        try:
            probe_pairs = await self.probe_gen.generate_probe_family(pair)
        except Exception as exc:
            log.warning("Counting unit %s as score=0: probe generation failed: %s", unit.id, exc)
            return failed_prediction_for_unit(unit, "probe_generation", exc)

        if not probe_pairs:
            return failed_prediction_for_unit(unit, "probe_generation", "No probes generated")

        fact = memory_to_fact(pair.original)
        evidence_trail: list[RoundEvidence] = []
        for round_idx, probe_batch in enumerate(group_probe_pairs_by_round(probe_pairs)):
            batch_fact_results: list[ProbeResult] = []
            batch_decoy_results: list[ProbeResult] = []

            for probe_f, probe_d in probe_batch:
                batch_fact_results.append(await self._execute_probe(agent, probe_f, access_level, target))
                batch_decoy_results.append(await self._execute_probe(agent, probe_d, access_level, target))

            probe_f, _ = probe_batch[0]
            features = await self._extract_round_features(
                pair,
                batch_fact_results,
                batch_decoy_results,
                probe_f.probe_type,
            )
            score = self.feat_ext.compute_round_score(features)
            evidence_trail.append(
                RoundEvidence(
                    fact_id=fact.id,
                    round_idx=round_idx,
                    probe_type=probe_f.probe_type,
                    score_fact=features.get("fact_similarity_mean", 0.0),
                    score_decoy=features.get("decoy_similarity_mean", 0.0),
                    delta_score=score,
                    features=features,
                    fact_results=batch_fact_results,
                    decoy_results=batch_decoy_results,
                )
            )

        final_score = (
            sum(e.delta_score for e in evidence_trail) / len(evidence_trail)
            if evidence_trail
            else 0.0
        )
        return MembershipPrediction(
            fact_id=fact.id,
            is_member_true=unit.is_member,
            score=final_score,
            is_member_pred=final_score > 0.0,
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
                    log.error("[MultiContrastive] unit %s failed: %s", unit.id, exc, exc_info=True)
                    pred = failed_prediction_for_unit(unit, "attack", exc)

                if idx % 10 == 0:
                    log.info("[MultiContrastive] %d/%d done", idx, len(units))
                return pred

        tasks = [attack_one(i, unit) for i, unit in enumerate(units, start=1)]
        return list(await asyncio.gather(*tasks))

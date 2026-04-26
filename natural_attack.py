"""CEA-MI attack using extracted MemoryUnit datasets and contrastive decoy pairs."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from agent_interface import AgentInterface
from config import Config
from decoy_builder import LLMDecoyBuilder
from evaluation import Evaluator
from evidence_accumulator import EvidenceAccumulator
from feature_extractor import FeatureExtractor
from llm_response_judge import LLMResponseJudge
from memory_extractor import MemoryExtractor
from memory_unit import MemoryDataset, MemoryUnit, MemoryUnitPair
from models import (
    AccessLevel,
    Fact,
    MembershipPrediction,
    ProbeResult,
    RoundEvidence,
)
from probe_batches import group_probe_pairs_by_round
from probe_generator import ProbeGenerator
from experiment_db import create_empty_memory_db, cleanup_isolated_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("natural_attack")


@dataclass
class AttackExample:
    """One contrastive attack example backed by a MemoryUnitPair."""

    user_id: int
    pair: MemoryUnitPair
    fact: Fact
    is_member: bool


def memory_to_fact(unit: MemoryUnit) -> Fact:
    """Adapt a MemoryUnit to the legacy Fact interface used by scoring code."""
    return Fact(
        id=unit.id,
        content=unit.content,
        topic=unit.topic or "",
        key_value=unit.key_value or "",
        category=unit.perlt_type.value,
        is_member=unit.is_member,
    )


def load_memory_dataset(path: Path) -> MemoryDataset:
    dataset = MemoryExtractor.load(path)
    if not dataset.users:
        raise ValueError(f"Loaded dataset has no users: {path}")
    if not dataset.all_members:
        raise ValueError(f"Loaded dataset has no member units: {path}")
    if not dataset.all_non_members:
        raise ValueError(f"Loaded dataset has no non-member units: {path}")
    return dataset


@dataclass
class UserAttackSet:
    """Attack samples for a single user."""
    user_id: int
    members: list[MemoryUnit]      # sampled member units to test
    non_members: list[MemoryUnit]  # sampled non-member units to test

    @property
    def all_units(self) -> list[MemoryUnit]:
        return self.members + self.non_members


def sample_attack_units_per_user(
    dataset: MemoryDataset,
    num_per_class: int,
    rng: random.Random,
    max_users: Optional[int] = None,
) -> list[UserAttackSet]:
    """Sample a balanced attack set for each user.

    For each user:
    - Sample min(num_per_class, available) members
    - Sample min(num_per_class, available) non-members

    Returns a list of UserAttackSet, one per user.
    """
    user_ids = dataset.user_ids
    if max_users is not None:
        user_ids = user_ids[:max_users]

    attack_sets: list[UserAttackSet] = []

    for user_id in user_ids:
        user_set = dataset.get_user(user_id)

        # Determine how many to sample for this user
        n_members = min(num_per_class, len(user_set.members))
        n_non_members = min(num_per_class, len(user_set.non_members))

        if n_members == 0 or n_non_members == 0:
            log.warning(
                "User %d has insufficient data (members=%d, non_members=%d), skipping",
                user_id, len(user_set.members), len(user_set.non_members)
            )
            continue

        sampled_members = rng.sample(user_set.members, n_members)
        sampled_non_members = rng.sample(user_set.non_members, n_non_members)

        attack_sets.append(UserAttackSet(
            user_id=user_id,
            members=sampled_members,
            non_members=sampled_non_members,
        ))

        log.info(
            "User %d: sampled %d members, %d non-members (from %d/%d available)",
            user_id, n_members, n_non_members,
            len(user_set.members), len(user_set.non_members)
        )

    if not attack_sets:
        raise ValueError("No users with sufficient data for attack")

    return attack_sets


class NaturalAttack:
    def __init__(
        self,
        cfg: Config,
        rng: random.Random,
        concurrency: int = 10,
        response_scorer: str = "rules",
    ):
        self.cfg = cfg
        self.rng = rng
        self.concurrency = concurrency
        self.response_scorer = response_scorer
        self._semaphore = asyncio.Semaphore(concurrency)
        self.agent = AgentInterface(
            api_base=cfg.api_base,
            api_key=cfg.api_key,
            model=cfg.model,
            db_path=cfg.nanobot_db_path,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
        self.decoy_builder = LLMDecoyBuilder(
            api_base=cfg.api_base,
            api_key=cfg.api_key,
            model=cfg.model,
            temperature=cfg.temperature,
        )
        self.probe_gen = ProbeGenerator(
            api_base=cfg.api_base,
            api_key=cfg.api_key,
            model=cfg.model,
            num_paraphrases=cfg.paraphrases_per_perspective,
        )
        self.feat_ext = FeatureExtractor()
        self.llm_judge = (
            LLMResponseJudge(
                api_base=cfg.api_base,
                api_key=cfg.api_key,
                model=cfg.model,
                temperature=0.0,
            )
            if response_scorer == "llm"
            else None
        )
        self.accumulator = EvidenceAccumulator(
            prior=cfg.prior,
            early_stop_threshold=1.0,
        )
        self.evaluator = Evaluator(bootstrap_n=cfg.bootstrap_n, seed=cfg.seed)
        self._prepared_user_id: Optional[int] = None
        self._prepared_memory_count: int = 0

    async def close(self):
        await self.cleanup()

    async def cleanup(self):
        await self.agent.close()
        await self.decoy_builder.close()
        await self.probe_gen.close()
        if self.llm_judge:
            await self.llm_judge.close()

    def _inject_user_memories(self, dataset: MemoryDataset, user_id: int):
        """Reset memory DB and inject the target user's member memories."""
        if self._prepared_user_id == user_id:
            return

        user_set = dataset.get_user(user_id)
        self.agent.clear_all_memory()

        injected = 0
        for unit in user_set.members:
            tags = [
                f"user:{unit.user_id}",
                f"type:{unit.perlt_type.value}",
                f"source:{unit.source_key}",
            ]
            if unit.topic:
                tags.append(f"topic:{unit.topic}")
            self.agent.inject_semantic_memory(unit.content, tags=tags)
            injected += 1

        self._prepared_user_id = user_id
        self._prepared_memory_count = injected
        log.info("Prepared user %s memory with %d member units", user_id, injected)

    async def _build_single_example(
        self, unit: MemoryUnit, idx: int, total: int
    ) -> Optional[AttackExample]:
        """Build a single attack example with semaphore-controlled concurrency."""
        async with self._semaphore:
            try:
                pair = await self.decoy_builder.build_decoy_pair_for_memory(unit)
            except Exception as exc:
                log.warning(
                    "Skipping unit %s (%d/%d): decoy pair generation failed: %s",
                    unit.id,
                    idx,
                    total,
                    exc,
                )
                return None

            if not pair.original.key_value or not pair.decoy.key_value:
                log.warning("Skipping unit %s: missing key_value in generated pair", unit.id)
                return None

            return AttackExample(
                user_id=unit.user_id,
                pair=pair,
                fact=memory_to_fact(pair.original),
                is_member=unit.is_member,
            )

    async def build_examples(self, units: list[MemoryUnit]) -> list[AttackExample]:
        """Generate decoy pairs and adapted Fact views for attack units (parallel)."""
        tasks = [
            self._build_single_example(unit, idx, len(units))
            for idx, unit in enumerate(units, start=1)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        examples: list[AttackExample] = []
        for r in results:
            if isinstance(r, AttackExample):
                examples.append(r)
            elif isinstance(r, Exception):
                log.warning("build_examples task failed: %s", r)
        return examples


    async def attack_user(
        self,
        dataset: MemoryDataset,
        user_attack_set: UserAttackSet,
        access_level: str,
    ) -> list[MembershipPrediction]:
        """Run the attack for a single user.

        1. Inject all of this user's member memories into agent
        2. Build decoy pairs for sampled units
        3. Probe each unit (both members and non-members)
        """
        user_id = user_attack_set.user_id
        log.info("=" * 50)
        log.info("Attacking User %d", user_id)
        log.info("=" * 50)

        # Step 1: Inject this user's member memories
        self._inject_user_memories(dataset, user_id)

        # Step 2: Build examples for this user's sampled units
        all_units = user_attack_set.all_units
        examples = await self.build_examples(all_units)

        if not examples:
            log.warning("No valid examples for user %d after decoy generation", user_id)
            return []

        ex_members = sum(1 for e in examples if e.is_member)
        ex_nonmembers = sum(1 for e in examples if not e.is_member)
        log.info(
            "User %d: %d examples ready (%d members, %d non-members)",
            user_id, len(examples), ex_members, ex_nonmembers
        )

        # Step 3: Probe each example (parallel with concurrency control)
        async def probe_one(idx: int, example: AttackExample) -> MembershipPrediction:
            log.info(
                "[User %d] [%d/%d] Probing unit=%s topic=%s kv=%s truth=%s",
                user_id,
                idx,
                len(examples),
                example.fact.id,
                example.fact.topic or "<none>",
                example.fact.key_value[:40] if example.fact.key_value else "<none>",
                example.is_member,
            )
            async with self._semaphore:
                try:
                    pred = await self._attack_single(example, access_level)
                    pred.is_member_true = example.is_member
                    log.info(
                        "  -> score=%.4f posterior=%.4f true=%s rounds=%d",
                        pred.score,
                        pred.posterior,
                        pred.is_member_true,
                        pred.num_rounds_used,
                    )
                    return pred
                except Exception as exc:
                    log.error("  FAILED: %s", exc, exc_info=True)
                    return MembershipPrediction(
                        fact_id=example.fact.id,
                        is_member_true=example.is_member,
                        posterior=0.5,
                        score=0.0,
                    )

        tasks = [probe_one(idx, ex) for idx, ex in enumerate(examples, start=1)]
        predictions = await asyncio.gather(*tasks)
        return list(predictions)

    async def attack_all_users(
        self,
        dataset: MemoryDataset,
        user_attack_sets: list[UserAttackSet],
        access_level: str,
        calibrate: bool = True,
    ) -> tuple[list[MembershipPrediction], dict[int, list[MembershipPrediction]]]:
        """Run the attack over all users.

        Returns:
            - all_predictions: Combined predictions from all users
            - per_user_predictions: Dict mapping user_id -> predictions for that user
        """
        # Calibration pass using first user's data
        if calibrate and user_attack_sets:
            log.info("--- Running calibration pass on first user ---")
            first_user = user_attack_sets[0]
            self._inject_user_memories(dataset, first_user.user_id)
            cal_units = first_user.members[:3] + first_user.non_members[:3]
            cal_examples = await self.build_examples(cal_units)
            if cal_examples:
                await self._calibration_pass_from_examples(cal_examples, access_level)

        all_predictions: list[MembershipPrediction] = []
        per_user_predictions: dict[int, list[MembershipPrediction]] = {}

        for user_idx, user_attack_set in enumerate(user_attack_sets, start=1):
            log.info(
                "\n[User %d/%d] Processing user_id=%d",
                user_idx, len(user_attack_sets), user_attack_set.user_id
            )
            user_preds = await self.attack_user(dataset, user_attack_set, access_level)
            per_user_predictions[user_attack_set.user_id] = user_preds
            all_predictions.extend(user_preds)

        return all_predictions, per_user_predictions

    async def _calibration_pass_from_examples(
        self,
        examples: list[AttackExample],
        access_level: str,
    ):
        """Calibrate using pre-built examples (memory already injected)."""
        member_examples = [e for e in examples if e.is_member]
        nonmember_examples = [e for e in examples if not e.is_member]

        if len(member_examples) < 1 or len(nonmember_examples) < 1:
            log.warning("Not enough calibration samples, skipping calibration")
            return

        member_scores: list[float] = []
        nonmember_scores: list[float] = []

        for example in examples:
            try:
                probe_pairs = await self.probe_gen.generate_probe_family(example.pair)
            except Exception as exc:
                log.warning("Calibration probe generation failed for %s: %s", example.fact.id, exc)
                continue

            round_scores = []
            for probe_f, probe_d in probe_pairs:
                result_f = await self._execute_probe(probe_f, access_level)
                result_d = await self._execute_probe(probe_d, access_level)
                features = await self._extract_round_features(
                    example,
                    [result_f],
                    [result_d],
                    probe_f.probe_type,
                )
                round_scores.append(self.feat_ext.compute_round_score(features))

            mean_score = sum(round_scores) / len(round_scores) if round_scores else 0.0
            if example.is_member:
                member_scores.append(mean_score)
            else:
                nonmember_scores.append(mean_score)

        if not member_scores or not nonmember_scores:
            log.warning("Calibration did not collect both member and non-member scores")
            return

        log.info("Calibration: member_scores=%s", [f"{s:.4f}" for s in member_scores])
        log.info("Calibration: nonmember_scores=%s", [f"{s:.4f}" for s in nonmember_scores])
        self.accumulator.calibrate_from_data(member_scores, nonmember_scores)
        log.info(
            "Calibrated distributions: member(%.4f, %.4f) nonmember(%.4f, %.4f)",
            self.accumulator.dist.member_mean,
            self.accumulator.dist.member_std,
            self.accumulator.dist.nonmember_mean,
            self.accumulator.dist.nonmember_std,
        )

    async def _attack_single(self, example: AttackExample, access_level: str) -> MembershipPrediction:
        probe_pairs = await self.probe_gen.generate_probe_family(example.pair)
        if not probe_pairs:
            log.warning("No probes generated for %s", example.fact.id)
            return MembershipPrediction(fact_id=example.fact.id, posterior=0.5, score=0.0)

        evidence_trail: list[RoundEvidence] = []
        for round_idx, probe_batch in enumerate(group_probe_pairs_by_round(probe_pairs)):
            batch_fact_results: list[ProbeResult] = []
            batch_decoy_results: list[ProbeResult] = []

            for probe_f, probe_d in probe_batch:
                batch_fact_results.append(await self._execute_probe(probe_f, access_level))
                batch_decoy_results.append(await self._execute_probe(probe_d, access_level))

            probe_f, _ = probe_batch[0]
            features = await self._extract_round_features(
                example,
                batch_fact_results,
                batch_decoy_results,
                probe_f.probe_type,
            )
            score = self.feat_ext.compute_round_score(features)

            log.info(
                "    round %d (%s): delta=%.4f sim_f=%.3f sim_d=%.3f mem_delta=%.3f",
                round_idx,
                probe_f.probe_type.value,
                score,
                features.get("fact_similarity_mean", 0.0),
                features.get("decoy_similarity_mean", 0.0),
                features.get("delta_memory_statement_score", 0.0),
            )

            evidence_trail.append(
                RoundEvidence(
                    fact_id=example.fact.id,
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

        pred = self.accumulator.accumulate(evidence_trail)
        if evidence_trail:
            pred.score = sum(e.delta_score for e in evidence_trail) / len(evidence_trail)
        return pred

    async def _extract_round_features(
        self,
        example: AttackExample,
        fact_results: list[ProbeResult],
        decoy_results: list[ProbeResult],
        probe_type,
    ) -> dict:
        if self.response_scorer == "rules":
            features = self.feat_ext.extract_round_features(
                example.fact,
                fact_results,
                decoy_results,
                probe_type,
            )
            features["response_scorer"] = "rules"
            return await self._add_whitebox_memory_statement_features(
                features,
                example,
                fact_results,
                decoy_results,
            )

        if self.response_scorer != "llm" or self.llm_judge is None:
            raise ValueError(f"Unknown response_scorer: {self.response_scorer}")

        fact_statement = example.pair.original.content
        decoy_statement = example.pair.decoy.content
        fact_topic = example.pair.original.topic or example.fact.topic or ""
        decoy_topic = example.pair.decoy.topic or fact_topic
        fact_key_value = example.pair.original.key_value or example.fact.key_value or ""
        decoy_key_value = example.pair.decoy.key_value or ""

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
            example,
            fact_results,
            decoy_results,
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
        example: AttackExample,
        fact_results: list[ProbeResult],
        decoy_results: list[ProbeResult],
    ) -> dict:
        if not any(r.memory_metadata is not None for r in fact_results + decoy_results):
            return features

        fact_statement = example.pair.original.content
        decoy_statement = example.pair.decoy.content
        fact_topic = example.pair.original.topic or example.fact.topic or ""
        decoy_topic = example.pair.decoy.topic or fact_topic
        fact_key_value = example.pair.original.key_value or example.fact.key_value or ""
        decoy_key_value = example.pair.decoy.key_value or ""

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

    async def _execute_probe(self, probe, access_level: str) -> ProbeResult:
        resp = await self.agent.query(probe.question, access_level=access_level)
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

    async def evaluate(
        self,
        predictions: list[MembershipPrediction],
        access_level: str,
        seed: int,
        output_dir: Path,
    ) -> dict:
        al = AccessLevel(access_level)
        report = self.evaluator.full_report(predictions, al, seed)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.evaluator.save_report(report, output_dir / "report.json")
        self.evaluator.save_predictions(predictions, output_dir / "predictions.json")
        return report


def compute_per_user_metrics(
    per_user_predictions: dict[int, list[MembershipPrediction]],
    evaluator: Evaluator,
    access_level: AccessLevel,
) -> dict:
    """Compute metrics for each user and aggregate."""
    import numpy as np

    user_metrics: dict[int, dict] = {}

    for user_id, preds in per_user_predictions.items():
        if len(preds) < 2:
            log.warning("User %d has too few predictions (%d), skipping metrics", user_id, len(preds))
            continue

        y_true = [int(p.is_member_true) for p in preds]
        if len(set(y_true)) < 2:
            log.warning("User %d has only one class in predictions, skipping metrics", user_id)
            continue

        report = evaluator.full_report(preds, access_level, seed=0)
        user_metrics[user_id] = report

    if not user_metrics:
        return {"error": "No users with valid metrics"}

    # Aggregate across users
    agg = {}
    for metric in ["accuracy", "brier_score", "ece"]:
        vals = [m.get(metric, 0.0) for m in user_metrics.values() if metric in m]
        if vals:
            agg[f"{metric}_mean"] = float(np.mean(vals))
            agg[f"{metric}_std"] = float(np.std(vals))

    for metric in ["roc_auc", "pr_auc"]:
        vals = []
        for m in user_metrics.values():
            v = m.get(metric)
            if isinstance(v, dict):
                vals.append(v.get("value", 0.0))
            elif isinstance(v, (int, float)):
                vals.append(float(v))
        if vals:
            agg[f"{metric}_mean"] = float(np.mean(vals))
            agg[f"{metric}_std"] = float(np.std(vals))

    agg["num_users"] = len(user_metrics)
    agg["per_user"] = user_metrics

    return agg


async def amain():
    parser = argparse.ArgumentParser(description="CEA-MI natural attack over MemoryUnit datasets")
    parser.add_argument("--access", choices=["blackbox", "graybox", "whitebox"], default="blackbox")
    parser.add_argument("--num-facts", type=int, default=10, help="attack units per class per user")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default="perltqa", help="Dataset name (perltqa, msqa, etc.)")
    parser.add_argument("--db", default=None, help="Nanobot memory DB path")
    parser.add_argument("--max-users", type=int, default=None, help="Optionally limit to the first N users")
    parser.add_argument("--multi-seed", action="store_true", help="Run seeds 42,123,456 and aggregate")
    parser.add_argument("--no-calibrate", action="store_true", help="Skip calibration pass")
    parser.add_argument("--concurrency", type=int, default=10, help="Max concurrent LLM calls")
    parser.add_argument(
        "--response-scorer",
        choices=["rules", "llm"],
        default="rules",
        help="Response scoring backend: rules uses local matching, llm uses an LLM judge",
    )
    parser.add_argument(
        "--num-paraphrases",
        type=int,
        default=0,
        help="Override the number of direct-recall paraphrases",
    )
    args = parser.parse_args()

    cfg = Config()
    if args.db:
        cfg.nanobot_db_path = Path(args.db)
    if args.num_paraphrases is not None:
        cfg.paraphrases_per_perspective = max(0, args.num_paraphrases)

    # Determine dataset path from name
    dataset_name = args.dataset
    dataset_path = cfg.data_dir / f"{dataset_name}_dialogue_seed42.json"

    if not dataset_path.exists():
        parser.error(f"Dataset file does not exist: {dataset_path}")

    dataset = load_memory_dataset(dataset_path)
    do_calibrate = not args.no_calibrate
    seeds = [42, 123, 456] if args.multi_seed else [args.seed]

    log.info("=" * 60)
    log.info("Dataset: %s", dataset_name)
    log.info("Loaded MemoryDataset from %s", dataset_path)
    log.info("Dataset stats: %s", dataset.stats())
    log.info("Using up to %s users", args.max_users if args.max_users is not None else "all")
    log.info("=" * 60)

    all_reports = []
    for seed in seeds:
        cfg.seed = seed
        rng = random.Random(seed)

        # Sample attack sets per user
        user_attack_sets = sample_attack_units_per_user(
            dataset,
            num_per_class=args.num_facts,
            rng=rng,
            max_users=args.max_users,
        )

        total_members = sum(len(uas.members) for uas in user_attack_sets)
        total_nonmembers = sum(len(uas.non_members) for uas in user_attack_sets)

        # Create isolated DB for this experiment run (enables parallel execution)
        algo_name = f"cea_mi_{args.response_scorer}"
        isolated_db = create_empty_memory_db(
            output_dir=cfg.output_dir,
            seed=seed,
            access_level=args.access,
            algo_name=algo_name,
        )
        cfg.nanobot_db_path = isolated_db
        log.info("Created isolated DB: %s", isolated_db)

        log.info("=" * 60)
        log.info("CEA-MI Natural Memory Attack (per-user, parallel)")
        log.info(
            "Access: %s | Units/class/user: %d | Seed: %d | Calibrate: %s | Concurrency: %d | Paraphrases: %d | Scorer: %s",
            args.access,
            args.num_facts,
            seed,
            do_calibrate,
            args.concurrency,
            cfg.paraphrases_per_perspective,
            args.response_scorer,
        )
        log.info("DB: %s", cfg.nanobot_db_path)
        log.info(
            "Attack set: %d users, %d members + %d nonmembers = %d total",
            len(user_attack_sets), total_members, total_nonmembers,
            total_members + total_nonmembers
        )
        log.info("=" * 60)

        attacker = NaturalAttack(
            cfg,
            rng,
            concurrency=args.concurrency,
            response_scorer=args.response_scorer,
        )
        start = time.time()
        try:
            # Run attack on all users
            all_predictions, per_user_predictions = await attacker.attack_all_users(
                dataset, user_attack_sets, args.access, calibrate=do_calibrate
            )

            if not all_predictions:
                raise RuntimeError("No predictions generated across all users")

            elapsed = time.time() - start
            output_name = f"{dataset_name}_{args.access}_{args.response_scorer}_seed{seed}"
            output_dir = cfg.output_dir / output_name

            # Global report (all predictions combined)
            global_report = await attacker.evaluate(all_predictions, args.access, seed, output_dir)

            # Per-user metrics
            per_user_metrics = compute_per_user_metrics(
                per_user_predictions,
                attacker.evaluator,
                AccessLevel(args.access),
            )

            log.info("=" * 60)
            log.info("RESULTS (seed=%d)", seed)
            log.info("=" * 60)

            # Global metrics
            log.info("--- Global (all users combined) ---")
            roc = global_report.get("roc_auc", {})
            if isinstance(roc, dict):
                log.info(
                    "ROC-AUC: %.4f (95%% CI: %.4f-%.4f)",
                    roc.get("value", 0.0),
                    roc.get("ci_lower", 0.0),
                    roc.get("ci_upper", 0.0),
                )
            pr = global_report.get("pr_auc", {})
            if isinstance(pr, dict):
                log.info("PR-AUC:  %.4f", pr.get("value", 0.0))
            log.info("Accuracy: %.4f", global_report.get("accuracy", 0.0))

            # Per-user averaged metrics
            log.info("--- Per-user averaged ---")
            log.info(
                "ROC-AUC: mean=%.4f std=%.4f",
                per_user_metrics.get("roc_auc_mean", 0.0),
                per_user_metrics.get("roc_auc_std", 0.0),
            )
            log.info(
                "Accuracy: mean=%.4f std=%.4f",
                per_user_metrics.get("accuracy_mean", 0.0),
                per_user_metrics.get("accuracy_std", 0.0),
            )
            log.info("Num users evaluated: %d", per_user_metrics.get("num_users", 0))

            log.info(
                "Avg rounds: %.2f  Early stop: %.1f%%",
                global_report.get("avg_rounds_used", 0.0),
                global_report.get("early_stop_rate", 0.0) * 100,
            )
            log.info("Queries:  %s", global_report.get("total_queries", "N/A"))
            log.info("Runtime:  %.1fs", elapsed)
            log.info("Output:   %s", output_dir)

            # Save metadata
            meta = {
                "version": "per_user_memoryunit_attack",
                "dataset_name": dataset_name,
                "dataset_path": str(dataset_path),
                "dataset_stats": dataset.stats(),
                "num_users": len(user_attack_sets),
                "total_members": total_members,
                "total_nonmembers": total_nonmembers,
                "runtime_seconds": elapsed,
                "calibration": do_calibrate,
                "response_scorer": args.response_scorer,
                "attack_protocol": "for each user: inject ALL member memories, then probe sampled members and non-members",
                "notes": [
                    "member/non-member labels come directly from MemoryDataset splits",
                    "each user's members are injected before testing that user's samples",
                    "non-members are never injected into agent memory",
                    "per_user_metrics shows metrics averaged across users",
                ],
            }
            with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            # Save per-user metrics
            with open(output_dir / "per_user_metrics.json", "w", encoding="utf-8") as f:
                json.dump(per_user_metrics, f, indent=2, default=str)

            # Combine for final report
            global_report["seed"] = seed
            global_report["per_user_metrics"] = per_user_metrics
            all_reports.append(global_report)
            print(json.dumps(global_report, indent=2, default=str))

            # Cleanup isolated DB after successful completion
            if cleanup_isolated_db(isolated_db):
                log.info("Cleaned up isolated DB: %s", isolated_db)

        finally:
            await attacker.cleanup()

    if len(all_reports) > 1:
        import numpy as np

        log.info("=" * 60)
        log.info("AGGREGATED RESULTS (%d seeds)", len(all_reports))
        log.info("=" * 60)
        for metric in ["accuracy", "brier_score", "ece"]:
            vals = [r.get(metric, 0.0) for r in all_reports]
            log.info("%s: mean=%.4f std=%.4f", metric, np.mean(vals), np.std(vals))
        for metric in ["roc_auc", "pr_auc"]:
            vals = [
                r.get(metric, {}).get("value", 0.0) if isinstance(r.get(metric), dict) else 0.0
                for r in all_reports
            ]
            log.info("%s: mean=%.4f std=%.4f", metric, np.mean(vals), np.std(vals))


if __name__ == "__main__":
    asyncio.run(amain())

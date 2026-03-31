"""CEA-MI attack adapted for embedding-based memory agent (MemGPT/Letta comparison).

This script reuses the CEA-MI framework but targets the EmbeddingMemoryAgent
instead of nanobot. The key difference is that embedding-based retrieval
creates a much clearer member/nonmember signal because:
  - Cosine similarity for member facts >> similarity for decoy facts
  - Semantic search naturally separates fact-specific from topic-generic queries

Usage:
    # First, set up the target agent:
    python setup_memgpt.py standalone --dataset /bigtemp/trv3px/benchmark_v2_dataset.json

    # Then run the attack:
    python memgpt_attack.py --access blackbox --num-facts 30 --seed 42

    # Multi-seed:
    python memgpt_attack.py --access blackbox --num-facts 30 --multi-seed
"""
from __future__ import annotations
import argparse
import asyncio
import json
import logging
import random
import sys
import time
import uuid
from pathlib import Path

# Add parent directory to path for CEA-MI imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from models import (Fact, DecoyPair, Probe, ProbeResult, ProbeType,
                    RoundEvidence, MembershipPrediction, AccessLevel)
from probe_generator import ProbeGenerator
from feature_extractor import FeatureExtractor
from evidence_accumulator import EvidenceAccumulator
from evaluation import Evaluator
from probe_batches import group_probe_pairs_by_round

from setup_memgpt import EmbeddingMemoryAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("memgpt_attack")

# Reuse decoy pools from natural_attack
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from natural_attack import (DECOY_VALUES, DEFAULT_DECOYS, load_dataset_facts,
                            build_decoy_fact, build_fact_objects)


def split_fact_pools(members, nonmembers, num_attack_per_class, calibration_per_class, rng):
    """Sample disjoint attack/calibration pools from the classified dataset."""
    attack_n = min(num_attack_per_class, len(members), len(nonmembers))
    if attack_n < num_attack_per_class:
        log.warning("Only %d per class available (requested %d)", attack_n, num_attack_per_class)

    if calibration_per_class <= 0:
        return (
            rng.sample(members, attack_n),
            rng.sample(nonmembers, attack_n),
            [],
            [],
        )

    required_members = attack_n + calibration_per_class
    required_nonmembers = attack_n + calibration_per_class
    if len(members) < required_members or len(nonmembers) < required_nonmembers:
        raise ValueError(
            "Not enough held-out facts to calibrate without leaking into evaluation. "
            f"Need at least {required_members} members and {required_nonmembers} nonmembers."
        )

    member_pool = rng.sample(members, required_members)
    nonmember_pool = rng.sample(nonmembers, required_nonmembers)
    attack_members = member_pool[:attack_n]
    cal_members = member_pool[attack_n:]
    attack_nonmembers = nonmember_pool[:attack_n]
    cal_nonmembers = nonmember_pool[attack_n:]
    return attack_members, attack_nonmembers, cal_members, cal_nonmembers


def classify_facts_embedding(dataset_facts, agent: EmbeddingMemoryAgent, threshold=0.4):
    """Classify facts using embedding similarity (ground truth from the agent's memory)."""
    members, nonmembers = [], []

    for df in dataset_facts:
        # Check if this fact has a close match in the embedding memory
        query = df["fact"]
        recalled = agent.recall(query, top_k=1, threshold=0.0)
        if recalled:
            best_sim = recalled[0]["similarity"]
        else:
            best_sim = 0.0

        df["match_score"] = best_sim
        if best_sim >= threshold:
            df["matched_memory"] = recalled[0]["content"] if recalled else ""
            members.append(df)
        else:
            nonmembers.append(df)

    log.info("Classified: %d members, %d nonmembers (threshold=%.2f)",
             len(members), len(nonmembers), threshold)
    return members, nonmembers


class MemGPTAttack:
    """CEA-MI attack targeting the embedding-based memory agent."""

    def __init__(self, cfg: Config, agent: EmbeddingMemoryAgent, rng: random.Random):
        self.cfg = cfg
        self.agent = agent
        self.rng = rng
        self.probe_gen = ProbeGenerator(
            api_base=cfg.api_base, api_key=cfg.api_key,
            model=cfg.model, num_paraphrases=cfg.paraphrases_per_perspective)
        self.feat_ext = FeatureExtractor()
        self.accumulator = EvidenceAccumulator(
            prior=cfg.prior,
            early_stop_threshold=1.0,
        )
        self.evaluator = Evaluator(bootstrap_n=cfg.bootstrap_n, seed=cfg.seed)

    async def _build_decoy_pairs(self, facts):
        return [DecoyPair(fact=f, decoy=build_decoy_fact(f, self.rng)) for f in facts]

    async def _collect_evidence_trail(self, pair, access_level):
        probe_pairs = await self.probe_gen.generate_probe_family(pair)
        if not probe_pairs:
            return []

        evidence_trail = []
        for round_idx, probe_batch in enumerate(group_probe_pairs_by_round(probe_pairs)):
            batch_fact_results = []
            batch_decoy_results = []
            for probe_f, probe_d in probe_batch:
                batch_fact_results.append(await self._execute_probe(probe_f, access_level))
                batch_decoy_results.append(await self._execute_probe(probe_d, access_level))

            probe_f, _ = probe_batch[0]
            features = self.feat_ext.extract_round_features(
                pair.fact, batch_fact_results, batch_decoy_results, probe_f.probe_type)
            score = self.feat_ext.compute_round_score(features)

            log.info("    round %d (%s): delta=%.4f sim_f=%.3f sim_d=%.3f",
                     round_idx, probe_f.probe_type.value, score,
                     features.get("fact_similarity_mean", 0),
                     features.get("decoy_similarity_mean", 0))

            evidence_trail.append(RoundEvidence(
                fact_id=pair.fact.id,
                round_idx=round_idx,
                probe_type=probe_f.probe_type,
                score_fact=features.get("fact_similarity_mean", 0),
                score_decoy=features.get("decoy_similarity_mean", 0),
                delta_score=score,
                features=features,
                fact_results=batch_fact_results,
                decoy_results=batch_decoy_results,
            ))
        return evidence_trail

    async def attack_all(self, facts, access_level):
        pairs = await self._build_decoy_pairs(facts)

        predictions = []
        total = len(pairs)
        for idx, pair in enumerate(pairs):
            log.info("[%d/%d] Probing: %s (kv=%s)", idx + 1, total,
                     pair.fact.content[:50], pair.fact.key_value[:20])
            try:
                pred = await self._attack_single(pair, access_level)
                pred.is_member_true = pair.fact.is_member
                predictions.append(pred)
                log.info("  -> score=%.4f true=%s", pred.score, pred.is_member_true)
            except Exception as e:
                log.error("  FAILED: %s", e, exc_info=True)
                predictions.append(MembershipPrediction(
                    fact_id=pair.fact.id, is_member_true=pair.fact.is_member,
                    posterior=0.5, score=0.5))
        return predictions

    async def _calibration_pass(self, pairs, access_level, n_cal=None):
        member_pairs = [p for p in pairs if p.fact.is_member]
        nonmem_pairs = [p for p in pairs if not p.fact.is_member]
        if n_cal is not None:
            member_pairs = member_pairs[:n_cal]
            nonmem_pairs = nonmem_pairs[:n_cal]
        if len(member_pairs) < 2 or len(nonmem_pairs) < 2:
            return

        member_scores, nonmember_scores = [], []
        for pair in member_pairs + nonmem_pairs:
            round_scores = [e.delta_score for e in await self._collect_evidence_trail(pair, access_level)]
            if pair.fact.is_member:
                member_scores.extend(round_scores)
            else:
                nonmember_scores.extend(round_scores)

        log.info("Calibration: member=%s nonmem=%s",
                 [f"{s:.4f}" for s in member_scores],
                 [f"{s:.4f}" for s in nonmember_scores])
        self.accumulator.calibrate_from_data(member_scores, nonmember_scores)

    async def _attack_single(self, pair, access_level):
        evidence_trail = await self._collect_evidence_trail(pair, access_level)
        if not evidence_trail:
            return MembershipPrediction(fact_id=pair.fact.id, posterior=0.5, score=0.5)

        pred = self.accumulator.accumulate(evidence_trail)
        if evidence_trail:
            pred.score = sum(e.delta_score for e in evidence_trail) / len(evidence_trail)
        return pred

    async def _execute_probe(self, probe, access_level):
        resp = await self.agent.query(probe.question, access_level=access_level)
        return ProbeResult(
            probe=probe, response=resp.get("response", ""),
            latency_ms=resp.get("latency_ms", 0),
            logprobs=resp.get("logprobs"),
            mean_logprob=resp.get("mean_logprob"),
            recall_triggered=resp.get("recall_triggered"),
            recall_top_similarity=resp.get("recall_top_similarity"),
            recall_hit_count=resp.get("recall_hit_count"))

    async def evaluate(self, predictions, access_level, seed, output_dir):
        al = AccessLevel(access_level)
        report = self.evaluator.full_report(predictions, al, seed)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.evaluator.save_report(report, output_dir / "report.json")
        self.evaluator.save_predictions(predictions, output_dir / "predictions.json")
        return report

    async def cleanup(self):
        await self.probe_gen.close()


async def amain():
    parser = argparse.ArgumentParser(description="CEA-MI MemGPT Attack")
    parser.add_argument("--access", choices=["blackbox", "graybox", "whitebox"], default="blackbox")
    parser.add_argument("--num-facts", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default="/bigtemp/trv3px/benchmark_v2_dataset.json")
    parser.add_argument("--memory-file", default="memgpt_memories.json")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--multi-seed", action="store_true")
    parser.add_argument("--no-calibrate", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    dataset_path = Path(args.dataset)
    do_calibrate = not args.no_calibrate
    seeds = [42, 123, 456] if args.multi_seed else [args.seed]

    # Load agent with pre-ingested memories
    agent = EmbeddingMemoryAgent(db_path=args.memory_file, vllm_base=cfg.api_base, vllm_model=cfg.model)

    # Load and classify
    dataset_facts = load_dataset_facts(dataset_path)
    members, nonmembers = classify_facts_embedding(dataset_facts, agent, args.threshold)

    if not members or not nonmembers:
        log.error("Need both members and nonmembers. members=%d, nonmembers=%d",
                  len(members), len(nonmembers))
        sys.exit(1)

    all_reports = []
    for seed in seeds:
        cfg.seed = seed
        rng = random.Random(seed)

        calibration_per_class = 5 if do_calibrate else 0
        run_calibration = do_calibrate
        calibration_facts = []
        try:
            attack_members, attack_nonmembers, cal_members, cal_nonmembers = split_fact_pools(
                members,
                nonmembers,
                num_attack_per_class=args.num_facts,
                calibration_per_class=calibration_per_class,
                rng=rng,
            )
        except ValueError as exc:
            log.warning("%s Skipping calibration for this run.", exc)
            attack_members, attack_nonmembers, cal_members, cal_nonmembers = split_fact_pools(
                members,
                nonmembers,
                num_attack_per_class=args.num_facts,
                calibration_per_class=0,
                rng=rng,
            )
            run_calibration = False

        facts = build_fact_objects(attack_members, attack_nonmembers, args.num_facts, rng)
        if cal_members or cal_nonmembers:
            calibration_facts = build_fact_objects(
                cal_members,
                cal_nonmembers,
                min(len(cal_members), len(cal_nonmembers)),
                rng,
            )
        else:
            run_calibration = False

        log.info("=" * 60)
        log.info("CEA-MI MemGPT Attack | Access: %s | Seed: %d | Calibrate: %s",
                 args.access, seed, run_calibration)
        log.info("=" * 60)

        start = time.time()
        attacker = MemGPTAttack(cfg, agent, rng)
        try:
            if run_calibration and calibration_facts:
                calibration_pairs = await attacker._build_decoy_pairs(calibration_facts)
                await attacker._calibration_pass(calibration_pairs, args.access)
            elif do_calibrate:
                log.warning("Calibration requested but no held-out facts were available; using default distributions")

            predictions = await attacker.attack_all(facts, args.access)
            output_dir = Path(cfg.output_dir) / ("memgpt_%s_seed%d" % (args.access, seed))
            report = await attacker.evaluate(predictions, args.access, seed, output_dir)
            elapsed = time.time() - start

            log.info("RESULTS (MemGPT, seed=%d)", seed)
            roc = report.get("roc_auc", {})
            if isinstance(roc, dict):
                log.info("ROC-AUC: %.4f", roc.get("value", 0))
            log.info("Accuracy: %.4f", report.get("accuracy", 0))
            log.info("Runtime: %.1fs", elapsed)

            all_reports.append(report)
            print(json.dumps(report, indent=2, default=str))
        finally:
            await attacker.cleanup()

    if len(all_reports) > 1:
        import numpy as np
        log.info("=" * 60)
        log.info("AGGREGATED (MemGPT, %d seeds)", len(all_reports))
        for m in ["accuracy"]:
            vals = [r.get(m, 0) for r in all_reports]
            log.info("  %s: mean=%.4f std=%.4f", m, np.mean(vals), np.std(vals))
        for m in ["roc_auc"]:
            vals = [r.get(m, {}).get("value", 0) if isinstance(r.get(m), dict) else 0 for r in all_reports]
            log.info("  %s: mean=%.4f std=%.4f", m, np.mean(vals), np.std(vals))


if __name__ == "__main__":
    asyncio.run(amain())

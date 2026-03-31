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
import os
import random
import sys
import time
import uuid
from pathlib import Path

# Add parent directory to path for CEA-MI imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config, DEFAULT_DATASET_PATH
from models import (Fact, DecoyPair, Probe, ProbeResult, ProbeType,
                    RoundEvidence, MembershipPrediction, AccessLevel)
from probe_generator import ProbeGenerator
from feature_extractor import FeatureExtractor
from evidence_accumulator import EvidenceAccumulator
from evaluation import Evaluator

from setup_memgpt import EmbeddingMemoryAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("memgpt_attack")

# Reuse decoy pools from natural_attack
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from natural_attack import (DECOY_VALUES, DEFAULT_DECOYS, load_dataset_facts,
                            build_decoy_fact, build_fact_objects)


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

    async def attack_all(self, facts, access_level, calibrate=True):
        pairs = []
        for f in facts:
            decoy = build_decoy_fact(f, self.rng)
            pairs.append(DecoyPair(fact=f, decoy=decoy))

        if calibrate:
            await self._calibration_pass(pairs, access_level)

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

    async def _calibration_pass(self, pairs, access_level, n_cal=5):
        member_pairs = [p for p in pairs if p.fact.is_member][:n_cal]
        nonmem_pairs = [p for p in pairs if not p.fact.is_member][:n_cal]
        if len(member_pairs) < 2 or len(nonmem_pairs) < 2:
            return

        member_scores, nonmember_scores = [], []
        for pair in member_pairs + nonmem_pairs:
            probe_pairs = await self.probe_gen.generate_probe_family(pair)
            round_scores = []
            for probe_f, probe_d in probe_pairs:
                result_f = await self._execute_probe(probe_f, access_level)
                result_d = await self._execute_probe(probe_d, access_level)
                features = self.feat_ext.extract_round_features(
                    pair.fact, [result_f], [result_d], probe_f.probe_type)
                round_scores.append(self.feat_ext.compute_round_score(features))
            mean_score = sum(round_scores) / len(round_scores) if round_scores else 0.0
            if pair.fact.is_member:
                member_scores.append(mean_score)
            else:
                nonmember_scores.append(mean_score)

        log.info("Calibration: member=%s nonmem=%s",
                 [f"{s:.4f}" for s in member_scores],
                 [f"{s:.4f}" for s in nonmember_scores])
        self.accumulator.calibrate_from_data(member_scores, nonmember_scores)

    async def _attack_single(self, pair, access_level):
        probe_pairs = await self.probe_gen.generate_probe_family(pair)
        if not probe_pairs:
            return MembershipPrediction(fact_id=pair.fact.id, posterior=0.5, score=0.5)

        evidence_trail = []
        for round_idx, (probe_f, probe_d) in enumerate(probe_pairs):
            result_f = await self._execute_probe(probe_f, access_level)
            result_d = await self._execute_probe(probe_d, access_level)
            features = self.feat_ext.extract_round_features(
                pair.fact, [result_f], [result_d], probe_f.probe_type)
            score = self.feat_ext.compute_round_score(features)

            log.info("    round %d (%s): delta=%.4f sim_f=%.3f sim_d=%.3f",
                     round_idx, probe_f.probe_type.value, score,
                     features.get("fact_similarity_mean", 0),
                     features.get("decoy_similarity_mean", 0))

            evidence = RoundEvidence(
                fact_id=pair.fact.id, round_idx=round_idx,
                probe_type=probe_f.probe_type,
                score_fact=features.get("fact_similarity_mean", 0),
                score_decoy=features.get("decoy_similarity_mean", 0),
                delta_score=score, features=features,
                fact_results=[result_f], decoy_results=[result_d])
            evidence_trail.append(evidence)

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
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH) if DEFAULT_DATASET_PATH else None)
    parser.add_argument(
        "--memory-file",
        default=os.environ.get(
            "CEA_MI_MEMGPT_MEMORY_FILE",
            str(Path(__file__).resolve().parent / "memgpt_memories.json"),
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--multi-seed", action="store_true")
    parser.add_argument("--no-calibrate", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    if not args.dataset:
        parser.error(
            "A benchmark dataset JSON is required. Pass --dataset /path/to/benchmark_v2_dataset.json "
            "or set CEA_MI_DATASET."
        )
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        parser.error(f"Dataset file does not exist: {dataset_path}")
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

        log.info("=" * 60)
        log.info("CEA-MI MemGPT Attack | Access: %s | Seed: %d", args.access, seed)
        log.info("=" * 60)

        facts = build_fact_objects(members, nonmembers, args.num_facts, rng)

        start = time.time()
        attacker = MemGPTAttack(cfg, agent, rng)
        try:
            predictions = await attacker.attack_all(facts, args.access, calibrate=do_calibrate)
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

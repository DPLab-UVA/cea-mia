"""CEA-MI attack adapted for Mem0-style embedding memory agent.

Mem0 uses embedding-based retrieval via vector stores. The system prompt
includes a "User Memories (from Mem0)" section that differs from MemGPT's
"Recalled Memories" section, testing whether CEA-MI generalizes across
different prompt injection styles for memory-augmented agents.

Usage:
    # First set up the target:
    python setup_mem0.py standalone --dataset /bigtemp/trv3px/benchmark_v2_dataset.json

    # Then run the attack:
    python mem0_attack.py --access blackbox --num-facts 30 --seed 42

    # All access levels:
    for access in blackbox graybox whitebox; do
        python mem0_attack.py --access $access --num-facts 30 --seed 42 \
            --memory-file mem0_memories.json \
            2>&1 | tee /bigtemp/trv3px/attack_mem0_${access}.log
    done
"""
from __future__ import annotations
import argparse
import asyncio
import json
import logging
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from models import (Fact, DecoyPair, Probe, ProbeResult, ProbeType,
                    RoundEvidence, MembershipPrediction, AccessLevel)
from probe_generator import ProbeGenerator
from feature_extractor import FeatureExtractor
from evidence_accumulator import EvidenceAccumulator
from evaluation import Evaluator
from natural_attack import (DECOY_VALUES, DEFAULT_DECOYS, load_dataset_facts,
                            build_decoy_fact, build_fact_objects)
from setup_mem0 import Mem0Agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("mem0_attack")


def classify_facts_from_split(dataset_facts, split_path: str):
    """Classify facts using the pre-computed member/nonmember split file."""
    with open(split_path, encoding="utf-8") as f:
        split = json.load(f)
    member_set = set(split["member_facts"])
    nonmember_set = set(split["nonmember_facts"])

    members, nonmembers = [], []
    for df in dataset_facts:
        raw = df.get("fact_raw", df.get("fact", ""))
        if raw in member_set:
            df["match_score"] = 1.0
            members.append(df)
        elif raw in nonmember_set:
            df["match_score"] = 0.0
            nonmembers.append(df)

    log.info("Classified from split: %d members, %d nonmembers", len(members), len(nonmembers))
    return members, nonmembers


class Mem0Attack:
    """CEA-MI attack targeting Mem0-style embedding memory agent."""

    def __init__(self, cfg: Config, agent: Mem0Agent, rng: random.Random):
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
    parser = argparse.ArgumentParser(description="CEA-MI Mem0 Attack")
    parser.add_argument("--access", choices=["blackbox", "graybox", "whitebox"], default="blackbox")
    parser.add_argument("--num-facts", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default="/bigtemp/trv3px/benchmark_v2_dataset.json")
    parser.add_argument("--memory-file", default="mem0_memories.json")
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--multi-seed", action="store_true")
    parser.add_argument("--no-calibrate", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    dataset_path = Path(args.dataset)
    do_calibrate = not args.no_calibrate
    seeds = [42, 123, 456] if args.multi_seed else [args.seed]

    agent = Mem0Agent(db_path=args.memory_file, vllm_base=cfg.api_base, vllm_model=cfg.model)

    dataset_facts = load_dataset_facts(dataset_path)
    split_file = args.split_file or str(Path(args.memory_file).with_suffix(".split.json"))
    members, nonmembers = classify_facts_from_split(dataset_facts, split_file)

    if not members or not nonmembers:
        log.error("Need both members and nonmembers. members=%d, nonmembers=%d",
                  len(members), len(nonmembers))
        sys.exit(1)

    all_reports = []
    for seed in seeds:
        cfg.seed = seed
        rng = random.Random(seed)

        log.info("=" * 60)
        log.info("CEA-MI Mem0 Attack | Access: %s | Seed: %d", args.access, seed)
        log.info("=" * 60)

        facts = build_fact_objects(members, nonmembers, args.num_facts, rng)

        start = time.time()
        attacker = Mem0Attack(cfg, agent, rng)
        try:
            predictions = await attacker.attack_all(facts, args.access, calibrate=do_calibrate)
            output_dir = Path(cfg.output_dir) / ("mem0_%s_seed%d" % (args.access, seed))
            report = await attacker.evaluate(predictions, args.access, seed, output_dir)
            elapsed = time.time() - start

            log.info("=" * 60)
            log.info("RESULTS (Mem0, seed=%d)", seed)
            log.info("=" * 60)
            roc = report.get("roc_auc", {})
            if isinstance(roc, dict):
                log.info("ROC-AUC: %.4f (95%% CI: %.4f-%.4f)",
                         roc.get("value", 0), roc.get("ci_lower", 0), roc.get("ci_upper", 0))
            log.info("Accuracy: %.4f", report.get("accuracy", 0))
            log.info("Runtime: %.1fs", elapsed)

            all_reports.append(report)
            print(json.dumps(report, indent=2, default=str))
        finally:
            await attacker.cleanup()

    if len(all_reports) > 1:
        import numpy as np
        log.info("=" * 60)
        log.info("AGGREGATED (Mem0, %d seeds)", len(all_reports))
        for m in ["accuracy"]:
            vals = [r.get(m, 0) for r in all_reports]
            log.info("  %s: mean=%.4f std=%.4f", m, np.mean(vals), np.std(vals))
        for m in ["roc_auc"]:
            vals = [r.get(m, {}).get("value", 0) if isinstance(r.get(m), dict) else 0
                    for r in all_reports]
            log.info("  %s: mean=%.4f std=%.4f", m, np.mean(vals), np.std(vals))


if __name__ == "__main__":
    asyncio.run(amain())

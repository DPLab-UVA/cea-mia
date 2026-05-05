"""CEA-MI attack adapted for embedding-based memory agent (MemGPT/Letta comparison).

This script reuses the CEA-MI framework but targets the EmbeddingMemoryAgent
instead of nanobot. The key difference is that embedding-based retrieval
creates a much clearer member/nonmember signal because:
  - Cosine similarity for member facts >> similarity for decoy facts
  - Semantic search naturally separates fact-specific from topic-generic queries

Usage:
    # First, set up the target agent:
    python setup_memgpt.py standalone --dataset data/perltqa_seed42.json

    # Then run the attack:
    python memgpt_attack.py --dataset data/perltqa_seed42.json --access blackbox --num-facts 30 --seed 42

    # Multi-seed:
    python memgpt_attack.py --dataset data/perltqa_seed42.json --access blackbox --num-facts 30 --multi-seed
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
from evaluation import Evaluator

from setup_memgpt import EmbeddingMemoryAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("memgpt_attack")

# Reuse decoy pools from natural_attack
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from natural_attack import (DECOY_VALUES, DEFAULT_DECOYS, load_dataset_facts,
                            build_decoy_fact, build_fact_objects)


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
        # facts not in either set are skipped

    log.info("Classified from split: %d members, %d nonmembers", len(members), len(nonmembers))
    return members, nonmembers


class MemGPTAttack:
    """CEA-MI attack targeting the embedding-based memory agent."""

    def __init__(self, cfg: Config, agent: EmbeddingMemoryAgent, rng: random.Random):
        self.cfg = cfg
        self.agent = agent
        self.rng = rng
        self.probe_gen = ProbeGenerator(
            api_base=cfg.api_base, api_key=cfg.api_key,
            model=cfg.model)
        self.feat_ext = FeatureExtractor()
        self.evaluator = Evaluator(bootstrap_n=cfg.bootstrap_n, seed=cfg.seed)

    async def attack_all(self, facts, access_level):
        pairs = []
        for f in facts:
            decoy = build_decoy_fact(f, self.rng)
            pairs.append(DecoyPair(fact=f, decoy=decoy))

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
                    score=0.0))
        return predictions

    async def _attack_single(self, pair, access_level):
        probe_pairs = await self.probe_gen.generate_probe_family(pair)
        if not probe_pairs:
            return MembershipPrediction(fact_id=pair.fact.id, score=0.0)

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

        final_score = (
            sum(e.delta_score for e in evidence_trail) / len(evidence_trail)
            if evidence_trail
            else 0.0
        )
        return MembershipPrediction(
            fact_id=pair.fact.id,
            score=final_score,
            is_member_pred=final_score > 0.0,
            evidence_trail=evidence_trail,
            num_rounds_used=len(evidence_trail),
        )

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
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--memory-file", default="memgpt_memories.json")
    parser.add_argument("--split-file", default=None,
                        help="Path to .split.json from setup_memgpt.py (default: memory-file with .split.json suffix)")
    parser.add_argument("--multi-seed", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    try:
        cfg.require_llm_config()
    except ValueError as exc:
        parser.error(str(exc))
    dataset_path = Path(args.dataset)
    seeds = [42, 123, 456] if args.multi_seed else [args.seed]

    # Load agent with pre-ingested memories
    agent = EmbeddingMemoryAgent(
        db_path=args.memory_file,
        vllm_base=cfg.api_base,
        vllm_model=cfg.model,
        vllm_api_key=cfg.api_key,
    )

    # Load and classify using the split file (ground truth)
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
        log.info("CEA-MI MemGPT Attack | Access: %s | Seed: %d", args.access, seed)
        log.info("=" * 60)

        facts = build_fact_objects(members, nonmembers, args.num_facts, rng)

        start = time.time()
        attacker = MemGPTAttack(cfg, agent, rng)
        try:
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

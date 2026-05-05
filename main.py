"""Main experiment runner for CEA-MI memory membership inference attack."""
from __future__ import annotations
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

from config import Config
from models import (Fact, DecoyPair, Probe, ProbeResult, ProbeType,
                    RoundEvidence, MembershipPrediction, AccessLevel)
from data_loader import DataLoader
from baselines.decoy_builder import DecoyBuilder
from probe_generator import ProbeGenerator
from agent_interface import AgentInterface
from feature_extractor import FeatureExtractor
from evaluation import Evaluator
from experiment_db import prepare_isolated_memory_db
from probe_batches import group_probe_pairs_by_round

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cea_mi")


class CEAMIExperiment:
    def __init__(self, config=None):
        self.cfg = config or Config()
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.data_dir.mkdir(parents=True, exist_ok=True)
        self.agent = AgentInterface(
            api_base=self.cfg.api_base, api_key=self.cfg.api_key,
            model=self.cfg.model, temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens)
        self.probe_gen = ProbeGenerator(
            api_base=self.cfg.api_base, api_key=self.cfg.api_key,
            model=self.cfg.model)
        self.feat_ext = FeatureExtractor()
        self.evaluator = Evaluator(bootstrap_n=self.cfg.bootstrap_n, seed=self.cfg.seed)

    def _prepare_isolated_memory_db(self, access_level: str, seed: int) -> Path:
        isolated_db_path = prepare_isolated_memory_db(
            source_db_path=self.cfg.nanobot_db_path,
            output_dir=self.cfg.output_dir,
            seed=seed,
            access_level=access_level,
        )
        self.agent.db_path = isolated_db_path
        self.agent._store = None
        self.agent._recall = None
        logger.info("Using isolated experiment DB copy: %s", isolated_db_path)
        return isolated_db_path

    async def setup_data(self, seed=None):
        s = seed or self.cfg.seed
        loader = DataLoader(seed=s, data_dir=self.cfg.data_dir)
        members, nonmembers = loader.generate_facts(
            self.cfg.num_member_facts, self.cfg.num_nonmember_facts)
        logger.info("Generated %d member + %d nonmember facts", len(members), len(nonmembers))
        self.agent.clear_all_memory()
        logger.info("Cleared agent memory")
        for fact in members:
            mid = self.agent.inject_semantic_memory(
                fact.content, tags=[fact.category, fact.topic.split("_")[0]])
            fact.memory_id = mid
            fact.is_member = True
        logger.info("Injected %d member facts into memory", len(members))
        all_facts = members + nonmembers
        all_pairs = loader.build_decoy_pairs(all_facts, all_facts)
        logger.info("Built %d decoy pairs", len(all_pairs))
        loader.save_dataset(members, nonmembers, all_pairs)
        return members, nonmembers, all_pairs

    async def run_attack(self, pairs, access_level="blackbox"):
        predictions = []
        total = len(pairs)
        for idx, pair in enumerate(pairs):
            logger.info("[%d/%d] Probing: %s...", idx + 1, total, pair.fact.content[:60])
            try:
                pred = await self._attack_single(pair, access_level)
                pred.is_member_true = pair.fact.is_member
                predictions.append(pred)
                logger.info("  score=%.3f true=%s rounds=%d",
                            pred.score, pred.is_member_true, pred.num_rounds_used)
            except Exception as e:
                logger.error("  FAILED: %s", e)
                predictions.append(MembershipPrediction(
                    fact_id=pair.fact.id, is_member_true=pair.fact.is_member,
                    score=0.0))
        return predictions

    async def _attack_single(self, pair, access_level):
        probe_pairs = await self.probe_gen.generate_probe_family(pair)
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
            evidence = RoundEvidence(
                fact_id=pair.fact.id, round_idx=round_idx,
                probe_type=probe_f.probe_type,
                score_fact=features.get("fact_similarity_mean", 0),
                score_decoy=features.get("decoy_similarity_mean", 0),
                delta_score=score, features=features,
                fact_results=batch_fact_results, decoy_results=batch_decoy_results)
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

    async def evaluate_results(self, predictions, access_level="blackbox", seed=0):
        al = AccessLevel(access_level)
        report = self.evaluator.full_report(predictions, al, seed)
        out_dir = self.cfg.output_dir / ("seed%d_%s" % (seed, access_level))
        self.evaluator.save_report(report, out_dir / "report.json")
        self.evaluator.save_predictions(predictions, out_dir / "predictions.json")
        return report

    async def run_full(self, access_level="blackbox", seed=None):
        s = seed or self.cfg.seed
        start = time.time()
        logger.info("=== CEA-MI Experiment: %s, seed=%d ===", access_level, s)
        self._prepare_isolated_memory_db(access_level, s)
        members, nonmembers, pairs = await self.setup_data(s)
        predictions = await self.run_attack(pairs, access_level)
        report = await self.evaluate_results(predictions, access_level, s)
        elapsed = time.time() - start
        report["runtime_seconds"] = elapsed
        logger.info("=== Results ===")
        roc = report.get("roc_auc", {})
        if isinstance(roc, dict):
            logger.info("ROC-AUC: %.4f", roc.get("value", 0))
        pr = report.get("pr_auc", {})
        if isinstance(pr, dict):
            logger.info("PR-AUC: %.4f", pr.get("value", 0))
        logger.info("Accuracy: %.4f", report.get("accuracy", 0))
        logger.info("Total queries: %s", report.get("total_queries", "N/A"))
        logger.info("Runtime: %.1fs", elapsed)
        return report

    async def cleanup(self):
        await self.agent.close()
        await self.probe_gen.close()


async def amain():
    import argparse
    parser = argparse.ArgumentParser(description="CEA-MI Memory Membership Inference Attack")
    parser.add_argument("--access", choices=["blackbox", "graybox", "whitebox"], default="blackbox")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--members", type=int, default=50)
    parser.add_argument("--nonmembers", type=int, default=50)
    args = parser.parse_args()
    cfg = Config()
    cfg.num_member_facts = args.members
    cfg.num_nonmember_facts = args.nonmembers
    cfg.seed = args.seed
    exp = CEAMIExperiment(cfg)
    try:
        report = await exp.run_full(access_level=args.access, seed=args.seed)
        print(json.dumps(report, indent=2, default=str))
    finally:
        await exp.cleanup()

if __name__ == "__main__":
    asyncio.run(amain())

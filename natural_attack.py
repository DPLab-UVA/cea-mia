"""CEA-MI attack using naturally-formed memories from nanobot's PMC pipeline."""
from __future__ import annotations
import argparse
import asyncio
import json
import logging
import random
import sqlite3
import sys
import time
import uuid
from pathlib import Path

from config import Config, DEFAULT_DATASET_PATH
from models import (Fact, DecoyPair, Probe, ProbeResult, ProbeType,
                    RoundEvidence, MembershipPrediction, AccessLevel)
from agent_interface import AgentInterface
from probe_generator import ProbeGenerator
from feature_extractor import FeatureExtractor
from evidence_accumulator import EvidenceAccumulator, DistributionParams
from evaluation import Evaluator
from probe_batches import group_probe_pairs_by_round

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("natural_attack")

# ── Decoy value pools for each topic category ────────────────────────────
DECOY_VALUES = {
    "name": ["Jordan", "Morgan", "Riley", "Casey", "Quinn"],
    "partner": ["Sarah", "Michael", "Emma", "David", "Olivia"],
    "pet": ["Buddy the golden retriever", "Luna the tabby cat", "Max the beagle"],
    "city": ["Portland", "Denver", "Austin", "Seattle", "Boston"],
    "birthday": ["March 22", "July 4", "November 11", "January 8", "September 30"],
    "job": ["data analyst", "backend developer", "product manager", "DevOps engineer"],
    "coffee": ["black coffee", "cappuccino", "green tea", "espresso", "chai latte"],
    "language": ["Java", "Go", "Ruby", "C++", "TypeScript"],
    "editor": ["Vim", "Emacs", "Sublime Text", "Atom", "IntelliJ"],
    "os": ["Windows", "Arch Linux", "Fedora", "ChromeOS"],
    "hobby": ["rock climbing", "watercolor painting", "chess", "running"],
    "music": ["jazz", "classical", "hip-hop", "country", "electronic"],
    "food": ["Thai", "Mexican", "Indian", "French", "Korean"],
    "framework": ["TensorFlow", "JAX", "Keras", "MXNet", "Caffe"],
    "gpu": ["RTX 4090", "A100 40GB", "V100 32GB", "H100 80GB"],
    "database": ["MongoDB", "MySQL", "Redis", "Cassandra"],
    "cloud": ["Azure VM", "GCP T4", "Oracle Cloud", "Paperspace"],
    "university": ["MIT", "Stanford", "CMU", "Georgia Tech", "UC Berkeley"],
    "meeting": ["Thursday", "Monday", "Friday", "Tuesday"],
    "restaurant": ["Nobu", "Olive Garden", "Shake Shack", "Panda Express"],
    "headphones": ["Bose QC45", "AirPods Max", "Sennheiser HD 660S"],
    "novel_defense": ["adversarial training", "input randomization", "model distillation"],
}
DEFAULT_DECOYS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "theta"]


# ── DB + Dataset loading ──────────────────────────────────────────────────

def load_semantic_memories(db_path: Path) -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    tables = [r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    log.info("DB tables: %s", tables)
    sem_table = None
    for cand in ("semantic_memories", "semantic_memory", "semantic"):
        if cand in tables:
            sem_table = cand
            break
    if sem_table is None:
        for t in tables:
            if "semantic" in t.lower():
                sem_table = t
                break
    if sem_table is None:
        log.error("No semantic table found in %s", tables)
        sys.exit(1)
    rows = cur.execute(f"SELECT * FROM {sem_table}").fetchall()
    memories = [dict(r) for r in rows]
    conn.close()
    log.info("Loaded %d semantic memories from '%s'", len(memories), sem_table)
    return memories


def get_mem_content(m: dict) -> str:
    for k in ("content", "text", "summary", "fact"):
        if k in m and m[k]:
            return str(m[k])
    return " ".join(str(v) for v in m.values() if isinstance(v, str) and len(str(v)) > 5)


def load_dataset_facts(path: Path) -> list[dict]:
    """Extract all facts_introduced from benchmark dataset."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    turns = data if isinstance(data, list) else data.get("turns", data.get("conversations", []))
    facts = []
    seen = set()
    for i, turn in enumerate(turns):
        meta = turn.get("metadata", {})
        introduced = meta.get("facts_introduced", [])
        if isinstance(introduced, str):
            introduced = [introduced]
        for fact_text in introduced:
            if not fact_text or not fact_text.strip():
                continue
            ft = fact_text.strip()
            if ft in seen:
                continue
            seen.add(ft)
            if "=" in ft:
                key, val = ft.split("=", 1)
                topic = key.strip().split("_")[0]  # first part for probe template matching
                key_value = val.strip()
                key_clean = key.strip().replace("_", " ")
                content = "Alex's %s is %s" % (key_clean, key_value)
            else:
                topic = ""
                key_value = ft
                content = ft
            facts.append({
                "turn_id": str(turn.get("id", turn.get("turn_id", i))),
                "fact_raw": ft,
                "fact": content,
                "key_value": key_value,
                "topic_raw": key.strip() if "=" in ft else "",
                "topic": topic,
                "category": turn.get("type", ""),
            })
    log.info("Extracted %d unique facts from %d turns", len(facts), len(turns))
    return facts


# ── Matching ──────────────────────────────────────────────────────────────

def tokenize(text: str) -> set[str]:
    return {w.strip(".,;:!?\"'()[]{}").lower() for w in text.split() if len(w) > 2}


def token_overlap(a: str, b: str) -> float:
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def classify_facts(dataset_facts, memories, threshold=0.20):
    """Split dataset facts into members (matched in DB) and nonmembers."""
    mem_contents = [(get_mem_content(m), m) for m in memories]
    members, nonmembers = [], []

    for df in dataset_facts:
        best_score, best_mem = 0.0, None
        # Try matching with content, key_value, and topic
        candidates = [df["fact"], df["key_value"]]
        if df.get("topic_raw"):
            candidates.append(df["topic_raw"].replace("_", " "))
        for mc, m_row in mem_contents:
            for cand in candidates:
                s = token_overlap(cand, mc)
                if s > best_score:
                    best_score = s
                    best_mem = mc
        df["match_score"] = best_score
        if best_score >= threshold:
            df["matched_memory"] = best_mem
            members.append(df)
        else:
            nonmembers.append(df)

    log.info("Classified: %d members, %d nonmembers (threshold=%.2f)",
             len(members), len(nonmembers), threshold)
    return members, nonmembers


# ── Build Fact + DecoyPair objects ────────────────────────────────────────

def build_fact_objects(members, nonmembers, num_per_class, rng):
    n = min(num_per_class, len(members), len(nonmembers))
    if n < num_per_class:
        log.warning("Only %d per class available (requested %d)", n, num_per_class)
    m_sample = rng.sample(members, n)
    nm_sample = rng.sample(nonmembers, n)

    facts = []
    for d in m_sample:
        facts.append(Fact(
            id="m_%s_%s" % (d["turn_id"], uuid.uuid4().hex[:4]),
            content=d["fact"],
            key_value=d["key_value"],
            topic=d.get("topic_raw", d.get("topic", "")),
            category=d.get("category", ""),
            is_member=True))
    for d in nm_sample:
        facts.append(Fact(
            id="nm_%s_%s" % (d["turn_id"], uuid.uuid4().hex[:4]),
            content=d["fact"],
            key_value=d["key_value"],
            topic=d.get("topic_raw", d.get("topic", "")),
            category=d.get("category", ""),
            is_member=False))
    rng.shuffle(facts)
    return facts


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


def build_decoy_fact(fact: Fact, rng: random.Random) -> Fact:
    """Build a plausible counterfactual decoy with proper key_value and topic."""
    topic_key = fact.topic.split("_")[0] if fact.topic else ""
    pool = DECOY_VALUES.get(topic_key, DEFAULT_DECOYS)
    # Pick a decoy value that's different from the real one
    candidates = [v for v in pool if v.lower() != fact.key_value.lower()]
    if not candidates:
        candidates = DEFAULT_DECOYS
    decoy_value = rng.choice(candidates)

    topic_clean = fact.topic.replace("_", " ") if fact.topic else "detail"
    decoy_content = "Alex's %s is %s" % (topic_clean, decoy_value)

    return Fact(
        id=fact.id + "_decoy",
        content=decoy_content,
        key_value=decoy_value,
        topic=fact.topic,
        category=fact.category,
        is_member=False)


# ── Attack logic ─────────────────────────────────────────────────────────

class NaturalAttack:
    def __init__(self, cfg: Config, rng: random.Random):
        self.cfg = cfg
        self.rng = rng
        self.agent = AgentInterface(
            api_base=cfg.api_base, api_key=cfg.api_key,
            model=cfg.model, temperature=cfg.temperature,
            max_tokens=cfg.max_tokens)
        self.probe_gen = ProbeGenerator(
            api_base=cfg.api_base, api_key=cfg.api_key,
            model=cfg.model, num_paraphrases=cfg.paraphrases_per_perspective)
        self.feat_ext = FeatureExtractor()
        # V3: early stopping disabled, default calibrated distributions
        self.accumulator = EvidenceAccumulator(
            prior=cfg.prior,
            early_stop_threshold=1.0,  # disabled — always run all rounds
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

            log.info("    round %d (%s): delta=%.4f  sim_f=%.3f sim_d=%.3f",
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

    async def _calibration_pass(self, pairs, access_level, n_cal=None):
        """Run held-out probes and calibrate on the same per-round score used at inference."""
        member_pairs = [p for p in pairs if p.fact.is_member]
        nonmem_pairs = [p for p in pairs if not p.fact.is_member]
        if n_cal is not None:
            member_pairs = member_pairs[:n_cal]
            nonmem_pairs = nonmem_pairs[:n_cal]
        if len(member_pairs) < 2 or len(nonmem_pairs) < 2:
            log.warning("Not enough calibration samples, skipping calibration")
            return

        member_scores, nonmember_scores = [], []
        for pair in member_pairs + nonmem_pairs:
            round_scores = [e.delta_score for e in await self._collect_evidence_trail(pair, access_level)]
            if pair.fact.is_member:
                member_scores.extend(round_scores)
            else:
                nonmember_scores.extend(round_scores)

        log.info("Calibration: member_scores=%s", [f"{s:.4f}" for s in member_scores])
        log.info("Calibration: nonmem_scores=%s", [f"{s:.4f}" for s in nonmember_scores])
        self.accumulator.calibrate_from_data(member_scores, nonmember_scores)
        log.info("Calibrated distributions: member(%.4f, %.4f) nonmember(%.4f, %.4f)",
                 self.accumulator.dist.member_mean, self.accumulator.dist.member_std,
                 self.accumulator.dist.nonmember_mean, self.accumulator.dist.nonmember_std)

    async def attack_all(self, facts, access_level):
        """Run attack on all facts, return list of MembershipPrediction."""
        pairs = await self._build_decoy_pairs(facts)

        predictions = []
        total = len(pairs)
        for idx, pair in enumerate(pairs):
            log.info("[%d/%d] Probing: %s (kv=%s, topic=%s)",
                     idx + 1, total, pair.fact.content[:50],
                     pair.fact.key_value[:20], pair.fact.topic)
            try:
                pred = await self._attack_single(pair, access_level)
                pred.is_member_true = pair.fact.is_member
                predictions.append(pred)
                log.info("  -> score=%.4f posterior=%.4f true=%s rounds=%d",
                         pred.score, pred.posterior, pred.is_member_true,
                         pred.num_rounds_used)
            except Exception as e:
                log.error("  FAILED: %s", e, exc_info=True)
                predictions.append(MembershipPrediction(
                    fact_id=pair.fact.id, is_member_true=pair.fact.is_member,
                    posterior=0.5, score=0.5))
        return predictions

    async def _attack_single(self, pair, access_level):
        evidence_trail = await self._collect_evidence_trail(pair, access_level)
        if not evidence_trail:
            log.warning("  No probes generated for %s", pair.fact.id)
            return MembershipPrediction(
                fact_id=pair.fact.id, posterior=0.5, score=0.5)

        pred = self.accumulator.accumulate(evidence_trail)
        # Also compute raw mean delta as an alternative score
        if evidence_trail:
            raw_mean = sum(e.delta_score for e in evidence_trail) / len(evidence_trail)
            # Use raw mean delta as the ROC score (more robust than LLR posterior)
            pred.score = raw_mean
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
        await self.agent.close()
        await self.probe_gen.close()


# ── Main ──────────────────────────────────────────────────────────────────

async def amain():
    parser = argparse.ArgumentParser(description="CEA-MI Natural Memory Attack")
    parser.add_argument("--access", choices=["blackbox", "graybox", "whitebox"], default="blackbox")
    parser.add_argument("--num-facts", type=int, default=30, help="facts per class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH) if DEFAULT_DATASET_PATH else None)
    parser.add_argument("--db", default=None, help="PMC db path (default: from config)")
    parser.add_argument("--threshold", type=float, default=0.20,
                        help="token overlap threshold for member classification")
    parser.add_argument("--multi-seed", action="store_true",
                        help="run across seeds 42,123,456 and aggregate")
    parser.add_argument("--no-calibrate", action="store_true",
                        help="skip calibration pass")
    args = parser.parse_args()
    if not args.dataset:
        parser.error(
            "A benchmark dataset JSON is required. Pass --dataset /path/to/benchmark_v2_dataset.json "
            "or set CEA_MI_DATASET."
        )

    cfg = Config()
    db_path = Path(args.db) if args.db else cfg.nanobot_db_path
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        parser.error(f"Dataset file does not exist: {dataset_path}")
    do_calibrate = not args.no_calibrate

    # Determine seeds
    seeds = [42, 123, 456] if args.multi_seed else [args.seed]

    # 1. Load and classify (shared across seeds)
    memories = load_semantic_memories(db_path)
    dataset_facts = load_dataset_facts(dataset_path)
    members, nonmembers = classify_facts(dataset_facts, memories, args.threshold)

    if not members:
        log.error("No members found — benchmark ingestion may have failed")
        sys.exit(1)
    if not nonmembers:
        log.error("All facts matched — try lowering --threshold")
        sys.exit(1)

    log.info("--- Member examples ---")
    for m in members[:3]:
        log.info("  %s -> matched: %s (score=%.3f)", m["fact_raw"], m.get("matched_memory","")[:60], m["match_score"])
    log.info("--- Nonmember examples ---")
    for n in nonmembers[:3]:
        log.info("  %s (best_score=%.3f)", n["fact_raw"], n["match_score"])

    calibration_per_class = 5 if do_calibrate else 0
    all_reports = []
    for seed in seeds:
        cfg.seed = seed
        rng = random.Random(seed)

        # 2. Sample disjoint attack/calibration pools
        calibration_facts = []
        run_calibration = do_calibrate
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
        log.info("CEA-MI Natural Memory Attack (v4 — confirmation probes + calibration)")
        log.info("Access: %s | Facts/class: %d | Seed: %d | Calibrate: %s",
                 args.access, args.num_facts, seed, run_calibration)
        log.info("DB: %s", db_path)
        log.info("=" * 60)

        n_mem = sum(1 for f in facts if f.is_member)
        n_nonmem = sum(1 for f in facts if not f.is_member)
        log.info("Attack set: %d members + %d nonmembers = %d total", n_mem, n_nonmem, len(facts))

        # 3. Run attack
        start = time.time()
        attacker = NaturalAttack(cfg, rng)
        try:
            if run_calibration and calibration_facts:
                log.info("--- Running calibration pass on %d held-out facts ---", len(calibration_facts))
                calibration_pairs = await attacker._build_decoy_pairs(calibration_facts)
                await attacker._calibration_pass(calibration_pairs, args.access)
            elif do_calibrate:
                log.warning("Calibration requested but no held-out facts were available; using default distributions")

            predictions = await attacker.attack_all(facts, args.access)
            output_dir = cfg.output_dir / ("natural_%s_seed%d_v4" % (args.access, seed))
            report = await attacker.evaluate(predictions, args.access, seed, output_dir)
            elapsed = time.time() - start

            # Print results
            log.info("=" * 60)
            log.info("RESULTS (v4, seed=%d)", seed)
            log.info("=" * 60)
            roc = report.get("roc_auc", {})
            if isinstance(roc, dict):
                log.info("ROC-AUC:  %.4f (95%% CI: %.4f-%.4f)",
                         roc.get("value", 0), roc.get("ci_lower", 0), roc.get("ci_upper", 0))
            pr = report.get("pr_auc", {})
            if isinstance(pr, dict):
                log.info("PR-AUC:   %.4f", pr.get("value", 0))
            log.info("Accuracy: %.4f", report.get("accuracy", 0))
            log.info("Brier:    %.4f", report.get("brier_score", 0))
            log.info("ECE:      %.4f", report.get("ece", 0))
            log.info("Avg rounds: %.2f  Early stop: %.1f%%",
                     report.get("avg_rounds_used", 0), report.get("early_stop_rate", 0) * 100)
            log.info("Queries:  %s", report.get("total_queries", "N/A"))
            log.info("Runtime:  %.1fs", elapsed)
            log.info("Output:   %s", output_dir)

            # Save metadata
            meta = {
                "version": "v4_confirmation_probes_calibration",
                "num_semantic_memories": len(memories),
                "num_dataset_facts": len(dataset_facts),
                "num_members_found": len(members),
                "num_nonmembers_found": len(nonmembers),
                "num_per_class_used": n_mem,
                "similarity_threshold": args.threshold,
                "runtime_seconds": elapsed,
                "calibration": run_calibration,
                "improvements": [
                    "confirmation probes (value-specific: 'My partner is Jamie, right?')",
                    "held-out calibration pass (5 per class, excluded from evaluation)",
                    "multi-seed support (--multi-seed for seeds 42,123,456)",
                    "delta_latency normalized (÷2000, clipped [-1,1])",
                    "delta_logprob normalized (÷5, clipped [-1,1])",
                    "early stopping disabled (threshold=1.0)",
                    "ROC score = raw mean delta (not LLR posterior)",
                ],
            }
            with open(output_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)

            report["seed"] = seed
            all_reports.append(report)
            print(json.dumps(report, indent=2, default=str))
        finally:
            await attacker.cleanup()

    # Aggregate multi-seed results
    if len(all_reports) > 1:
        log.info("=" * 60)
        log.info("AGGREGATED RESULTS (v4, %d seeds)", len(all_reports))
        log.info("=" * 60)
        import numpy as _np
        for metric in ["accuracy", "brier_score", "ece"]:
            vals = [r.get(metric, 0) for r in all_reports]
            log.info("  %s: mean=%.4f std=%.4f", metric, _np.mean(vals), _np.std(vals))
        for metric in ["roc_auc", "pr_auc"]:
            vals = [r.get(metric, {}).get("value", 0) if isinstance(r.get(metric), dict) else 0 for r in all_reports]
            log.info("  %s: mean=%.4f std=%.4f", metric, _np.mean(vals), _np.std(vals))
        agg = {"seeds": seeds, "access_level": args.access, "per_seed": all_reports}
        agg_dir = cfg.output_dir / ("natural_%s_aggregate_v4" % args.access)
        agg_dir.mkdir(parents=True, exist_ok=True)
        with open(agg_dir / "aggregate_report.json", "w") as f:
            json.dump(agg, f, indent=2, default=str)


if __name__ == "__main__":
    asyncio.run(amain())

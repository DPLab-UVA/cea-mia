"""Baseline MIA attacks for comparison with CEA-MI.

Implements four standard membership inference baselines:
  1. Naive Single Query (no contrastive, no multi-round)
  2. Loss/Perplexity MIA (Yeom et al., 2018)
  3. Min-K% Prob (Shi et al., 2024)
  4. Reference Model Attack (Carlini et al., 2022)

Usage:
    python baseline_attacks.py --target nanobot --access blackbox --num-facts 30 --seed 42
    python baseline_attacks.py --target memgpt --access blackbox --num-facts 30 --seed 42
    python baseline_attacks.py --target mem0 --access blackbox --num-facts 30 --seed 42
"""
from __future__ import annotations
import argparse
import asyncio
import json
import logging
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from models import Fact, MembershipPrediction, AccessLevel
from evaluation import Evaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("baselines")


# ── Target agent loaders ─────────────────────────────────────────────────

def load_target_agent(target: str, memory_file: str, cfg: Config):
    """Load the appropriate agent based on target type."""
    if target == "memgpt":
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "memgpt_target"))
        from setup_memgpt import EmbeddingMemoryAgent
        return EmbeddingMemoryAgent(db_path=memory_file,
                                    vllm_base=cfg.api_base, vllm_model=cfg.model)
    elif target == "mem0":
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mem0_target"))
        from setup_mem0 import Mem0Agent
        return Mem0Agent(db_path=memory_file,
                         vllm_base=cfg.api_base, vllm_model=cfg.model)
    elif target == "nanobot":
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from agent_interface import AgentInterface
        return AgentInterface(api_base=cfg.api_base, api_key=cfg.api_key,
                              model=cfg.model, temperature=cfg.temperature,
                              max_tokens=cfg.max_tokens)
    else:
        raise ValueError(f"Unknown target: {target}")


def load_facts_and_split(target: str, dataset_path: str, memory_file: str,
                         db_path: str, num_facts: int, seed: int):
    """Load facts with ground truth labels."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from natural_attack import load_dataset_facts, build_fact_objects

    dataset_facts = load_dataset_facts(Path(dataset_path))
    rng = random.Random(seed)

    if target == "nanobot":
        from natural_attack import load_semantic_memories, classify_facts
        memories = load_semantic_memories(Path(db_path))
        members, nonmembers = classify_facts(dataset_facts, memories, 0.20)
    else:
        split_file = str(Path(memory_file).with_suffix(".split.json"))
        with open(split_file, encoding="utf-8") as f:
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

    facts = build_fact_objects(members, nonmembers, num_facts, rng)
    log.info("Loaded %d facts (%d members, %d nonmembers)",
             len(facts), sum(1 for f in facts if f.is_member),
             sum(1 for f in facts if not f.is_member))
    return facts


# ── Shared query helper ──────────────────────────────────────────────────

async def query_agent(agent, message: str, access_level: str, target: str) -> dict:
    """Unified query interface across targets."""
    if target == "nanobot":
        return await agent.query(message, access_level=access_level)
    else:
        return await agent.query(message, access_level=access_level)


# ═══════════════════════════════════════════════════════════════════════
# Baseline 1: Naive Single Query
# ═══════════════════════════════════════════════════════════════════════

class NaiveSingleQuery:
    """Ask one direct question per fact. Score = text similarity of response to key_value.

    No contrastive design (no decoy), no multi-round evidence accumulation.
    This is what a naive attacker would try first.
    """
    name = "naive_single_query"

    @staticmethod
    def text_similarity(response: str, expected: str) -> float:
        if not expected or not response:
            return 0.0
        rl, el = response.lower(), expected.lower()
        if el in rl:
            return 1.0
        rt, et = set(rl.split()), set(el.split())
        return len(rt & et) / len(et) if et else 0.0

    async def attack(self, agent, facts, access_level, target) -> list[MembershipPrediction]:
        predictions = []
        for i, fact in enumerate(facts):
            topic = fact.topic.split("_")[0] if fact.topic else "detail"
            question = f"What do you know about my {topic}?"

            resp = await query_agent(agent, question, access_level, target)
            sim = self.text_similarity(resp.get("response", ""), fact.key_value)

            predictions.append(MembershipPrediction(
                fact_id=fact.id, is_member_true=fact.is_member,
                score=sim, posterior=sim, is_member_pred=sim > 0.3,
                num_rounds_used=1))

            if (i + 1) % 10 == 0:
                log.info("[NaiveSingle] %d/%d done", i + 1, len(facts))

        return predictions


# ═══════════════════════════════════════════════════════════════════════
# Baseline 2: Loss/Perplexity MIA (Yeom et al., 2018)
# ═══════════════════════════════════════════════════════════════════════

class LossBasedMIA:
    """Feed the fact text as a completion and measure average logprob.

    Members should have lower perplexity (higher logprob) because the
    model has seen them in its memory context.

    Requires graybox access (logprobs).
    """
    name = "loss_mia"

    async def attack(self, agent, facts, access_level, target) -> list[MembershipPrediction]:
        predictions = []
        for i, fact in enumerate(facts):
            # Ask the model to complete/confirm the fact
            prompt = f"Complete this fact about the user: {fact.content}"
            resp = await query_agent(agent, prompt, access_level, target)

            mean_lp = resp.get("mean_logprob")
            if mean_lp is not None:
                # Higher logprob = more likely member
                # Normalize to [0,1] via sigmoid
                score = 1.0 / (1.0 + math.exp(-mean_lp - 2.0))
            else:
                score = 0.5  # no logprob available

            predictions.append(MembershipPrediction(
                fact_id=fact.id, is_member_true=fact.is_member,
                score=score, posterior=score, is_member_pred=score > 0.5,
                num_rounds_used=1))

            if (i + 1) % 10 == 0:
                log.info("[LossMIA] %d/%d done", i + 1, len(facts))

        return predictions


# ═══════════════════════════════════════════════════════════════════════
# Baseline 3: Min-K% Prob (Shi et al., 2024)
# ═══════════════════════════════════════════════════════════════════════

class MinKProbMIA:
    """Use the average of the lowest k% token logprobs as membership signal.

    Intuition: for members, even the least-likely tokens have higher
    probability because the model is more confident about memorized content.

    Requires graybox access (logprobs).
    """
    name = "min_k_prob"

    def __init__(self, k_pct: float = 0.2):
        self.k_pct = k_pct

    async def attack(self, agent, facts, access_level, target) -> list[MembershipPrediction]:
        predictions = []
        for i, fact in enumerate(facts):
            prompt = f"Tell me about: {fact.content}"
            resp = await query_agent(agent, prompt, access_level, target)

            logprobs = resp.get("logprobs", [])
            if logprobs and len(logprobs) > 0:
                k = max(1, int(len(logprobs) * self.k_pct))
                sorted_lps = sorted(logprobs)
                min_k_avg = sum(sorted_lps[:k]) / k
                # Higher min-k avg = more likely member
                score = 1.0 / (1.0 + math.exp(-min_k_avg - 3.0))
            else:
                score = 0.5

            predictions.append(MembershipPrediction(
                fact_id=fact.id, is_member_true=fact.is_member,
                score=score, posterior=score, is_member_pred=score > 0.5,
                num_rounds_used=1))

            if (i + 1) % 10 == 0:
                log.info("[MinK] %d/%d done", i + 1, len(facts))

        return predictions


# ═══════════════════════════════════════════════════════════════════════
# Baseline 4: Reference Model Attack (Carlini et al., 2022)
# ═══════════════════════════════════════════════════════════════════════

class ReferenceModelMIA:
    """Compare target model (with memory) vs reference (without memory).

    Score = logprob_with_memory - logprob_without_memory.
    Positive = memory helped = likely member.

    The 'reference' is the same LLM but queried without memory context.
    Requires graybox access.
    """
    name = "reference_model"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    async def _query_no_memory(self, message: str) -> dict:
        """Query the raw LLM without any memory augmentation."""
        import httpx
        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": message},
            ],
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "logprobs": True,
            "top_logprobs": 5,
        }
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{self.cfg.api_base}/chat/completions",
                headers={"Authorization": f"Bearer {self.cfg.api_key}"},
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        choice = data["choices"][0]
        logprobs_data = choice.get("logprobs", {})
        token_logprobs = []
        if logprobs_data and logprobs_data.get("content"):
            token_logprobs = [t["logprob"] for t in logprobs_data["content"]]

        return {
            "response": choice["message"]["content"],
            "mean_logprob": sum(token_logprobs) / len(token_logprobs) if token_logprobs else None,
        }

    async def attack(self, agent, facts, access_level, target) -> list[MembershipPrediction]:
        predictions = []
        for i, fact in enumerate(facts):
            topic = fact.topic.split("_")[0] if fact.topic else "detail"
            question = f"What do you know about my {topic}?"

            # Query WITH memory
            resp_mem = await query_agent(agent, question, access_level, target)
            # Query WITHOUT memory (raw LLM)
            resp_ref = await self._query_no_memory(question)

            lp_mem = resp_mem.get("mean_logprob")
            lp_ref = resp_ref.get("mean_logprob")

            if lp_mem is not None and lp_ref is not None:
                # Positive delta = memory boosted confidence = likely member
                delta = lp_mem - lp_ref
                score = 1.0 / (1.0 + math.exp(-delta * 5.0))
            else:
                score = 0.5

            predictions.append(MembershipPrediction(
                fact_id=fact.id, is_member_true=fact.is_member,
                score=score, posterior=score, is_member_pred=score > 0.5,
                num_rounds_used=1))

            if (i + 1) % 10 == 0:
                log.info("[RefModel] %d/%d done", i + 1, len(facts))

        return predictions


# ═══════════════════════════════════════════════════════════════════════
# Baseline 5: Multi-Query No Contrastive
# ═══════════════════════════════════════════════════════════════════════

class MultiQueryNoContrastive:
    """Ask multiple probe types (like CEA-MI) but WITHOUT contrastive decoy.

    Score = average text similarity across probe types.
    This isolates CEA-MI's contrastive contribution.
    """
    name = "multi_query_no_contrastive"

    @staticmethod
    def text_similarity(response: str, expected: str) -> float:
        if not expected or not response:
            return 0.0
        rl, el = response.lower(), expected.lower()
        if el in rl:
            return 1.0
        rt, et = set(rl.split()), set(el.split())
        return len(rt & et) / len(et) if et else 0.0

    async def attack(self, agent, facts, access_level, target) -> list[MembershipPrediction]:
        predictions = []
        for i, fact in enumerate(facts):
            topic = fact.topic.split("_")[0] if fact.topic else "detail"
            questions = [
                f"What do you know about my {topic}?",
                f"Can you remind me about my {topic}?",
                f"Help me with something related to {topic}, using what you know about me.",
                f"I think my {topic} might be different from what we discussed. What's correct?",
                f"What do you know about my {topic} and where did that info come from?",
            ]

            sims = []
            for q in questions:
                resp = await query_agent(agent, q, access_level, target)
                sim = self.text_similarity(resp.get("response", ""), fact.key_value)
                sims.append(sim)

            score = sum(sims) / len(sims)
            predictions.append(MembershipPrediction(
                fact_id=fact.id, is_member_true=fact.is_member,
                score=score, posterior=score, is_member_pred=score > 0.3,
                num_rounds_used=len(questions)))

            if (i + 1) % 10 == 0:
                log.info("[MultiNoContrast] %d/%d done", i + 1, len(facts))

        return predictions


# ── Main ──────────────────────────────────────────────────────────────────

BASELINES = {
    "naive": NaiveSingleQuery,
    "loss": LossBasedMIA,
    "mink": MinKProbMIA,
    "reference": ReferenceModelMIA,
    "multi_no_contrastive": MultiQueryNoContrastive,
}

# Which baselines work for which access levels
BASELINE_ACCESS = {
    "naive": ["blackbox", "graybox", "whitebox"],
    "loss": ["graybox", "whitebox"],
    "mink": ["graybox", "whitebox"],
    "reference": ["graybox", "whitebox"],
    "multi_no_contrastive": ["blackbox", "graybox", "whitebox"],
}


async def run_baseline(baseline_name, agent, facts, access_level, target, cfg):
    if baseline_name == "reference":
        attack = ReferenceModelMIA(cfg)
    elif baseline_name == "mink":
        attack = MinKProbMIA(k_pct=0.2)
    else:
        attack = BASELINES[baseline_name]()

    log.info("Running baseline: %s (target=%s, access=%s, n=%d)",
             baseline_name, target, access_level, len(facts))
    start = time.time()
    predictions = await attack.attack(agent, facts, access_level, target)
    elapsed = time.time() - start
    log.info("Baseline %s completed in %.1fs", baseline_name, elapsed)
    return predictions, elapsed


async def amain():
    parser = argparse.ArgumentParser(description="Baseline MIA attacks")
    parser.add_argument("--target", choices=["nanobot", "memgpt", "mem0"], required=True)
    parser.add_argument("--access", choices=["blackbox", "graybox", "whitebox"], default="blackbox")
    parser.add_argument("--num-facts", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default="/bigtemp/trv3px/benchmark_v2_dataset.json")
    parser.add_argument("--memory-file", default=None)
    parser.add_argument("--db", default=None)
    parser.add_argument("--baseline", default="all",
                        help="Which baseline: naive,loss,mink,reference,multi_no_contrastive,all")
    args = parser.parse_args()

    cfg = Config()
    cfg.seed = args.seed

    # Default memory files per target
    if args.memory_file is None:
        if args.target == "memgpt":
            args.memory_file = "memgpt_memories.json"
        elif args.target == "mem0":
            args.memory_file = "mem0_memories.json"

    db_path = args.db or str(cfg.nanobot_db_path)

    # Load target agent
    agent = load_target_agent(args.target, args.memory_file, cfg)

    # Load facts
    facts = load_facts_and_split(args.target, args.dataset, args.memory_file,
                                 db_path, args.num_facts, args.seed)

    # Determine which baselines to run
    if args.baseline == "all":
        baseline_names = [b for b, levels in BASELINE_ACCESS.items()
                          if args.access in levels]
    else:
        baseline_names = [b.strip() for b in args.baseline.split(",")]

    evaluator = Evaluator(bootstrap_n=1000, seed=args.seed)
    all_results = {}

    for bname in baseline_names:
        if args.access not in BASELINE_ACCESS.get(bname, []):
            log.warning("Skipping %s (requires %s, have %s)",
                        bname, BASELINE_ACCESS[bname], args.access)
            continue

        predictions, elapsed = await run_baseline(
            bname, agent, facts, args.access, args.target, cfg)

        al = AccessLevel(args.access)
        report = evaluator.full_report(predictions, al, args.seed)

        # Save results
        output_dir = Path(cfg.output_dir) / f"baseline_{bname}_{args.target}_{args.access}_seed{args.seed}"
        output_dir.mkdir(parents=True, exist_ok=True)
        evaluator.save_report(report, output_dir / "report.json")
        evaluator.save_predictions(predictions, output_dir / "predictions.json")

        roc = report.get("roc_auc", {})
        roc_val = roc.get("value", 0) if isinstance(roc, dict) else 0
        acc = report.get("accuracy", 0)

        all_results[bname] = {
            "roc_auc": roc_val, "accuracy": acc,
            "runtime": elapsed, "queries": len(facts),
        }

        log.info("[%s] ROC-AUC=%.4f  Accuracy=%.4f  Runtime=%.1fs",
                 bname, roc_val, acc, elapsed)

    # Print comparison table
    log.info("=" * 60)
    log.info("BASELINE COMPARISON (target=%s, access=%s)", args.target, args.access)
    log.info("=" * 60)
    log.info("%-25s  ROC-AUC  Accuracy  Queries  Runtime", "Method")
    log.info("-" * 60)
    for bname, r in all_results.items():
        log.info("%-25s  %.4f   %.4f    %d      %.1fs",
                 bname, r["roc_auc"], r["accuracy"], r["queries"], r["runtime"])
    log.info("-" * 60)

    # Save combined results
    combined_dir = Path(cfg.output_dir) / f"baselines_{args.target}_{args.access}_seed{args.seed}"
    combined_dir.mkdir(parents=True, exist_ok=True)
    with open(combined_dir / "comparison.json", "w") as f:
        json.dump({"target": args.target, "access_level": args.access,
                    "seed": args.seed, "results": all_results}, f, indent=2)

    # Cleanup
    if hasattr(agent, "close"):
        await agent.close()


if __name__ == "__main__":
    asyncio.run(amain())

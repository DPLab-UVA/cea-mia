"""CEA-MI natural attack entrypoint using direct multi-probe recall by default."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import time
from pathlib import Path
from typing import Optional

from config import Config
from evaluation import Evaluator
from experiment_db import cleanup_isolated_db, create_empty_memory_db
from memory_attack_utils import (
    ACCESS_DERIVATION_ORDER,
    UserAttackSet,
    compute_per_user_metrics,
    compute_probe_type_ablation,
    compute_top_n_response_ablation,
    failed_prediction_for_unit,
    load_memory_dataset,
    memory_to_fact,
    parse_optional_num_facts,
    project_per_user_predictions_for_access,
    project_predictions_for_access,
    resolve_memory_dataset_path,
    sample_attack_units_per_user,
)
from memory_unit import MemoryDataset
from models import AccessLevel, MembershipPrediction
from multi_probe_attack import MultiProbeDirectAttack
from target_adapters import TARGET_CHOICES, load_target_agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("natural_attack")


class NaturalAttack:
    """Default natural attack: no decoy, k direct recall probes per memory."""

    def __init__(
        self,
        cfg: Config,
        rng: random.Random,
        concurrency: int = 40,
        response_scorer: str = "rules",
        target: str = "nanobot",
        memory_file: Optional[Path] = None,
        direct_probe_k: int = 5,
    ):
        self.cfg = cfg
        self.rng = rng
        self.concurrency = concurrency
        self.response_scorer = response_scorer
        self.target = target
        self.memory_file = Path(memory_file) if memory_file else None
        self.direct_probe_k = max(1, int(direct_probe_k))
        self.agent = load_target_agent(
            target,
            cfg,
            db_path=cfg.nanobot_db_path,
            memory_file=self.memory_file,
        )
        self.multi_probe = MultiProbeDirectAttack(
            api_base=cfg.api_base,
            api_key=cfg.api_key,
            model=cfg.model,
            temperature=cfg.temperature,
            probe_concurrency=concurrency,
            response_scorer=response_scorer,
            direct_probe_k=self.direct_probe_k,
            memory_statement_judge=True,
        )
        self.evaluator = Evaluator(bootstrap_n=cfg.bootstrap_n, seed=cfg.seed)
        self._prepared_user_id: Optional[int] = None
        self._prepared_memory_count: int = 0

    async def close(self):
        await self.cleanup()

    async def cleanup(self):
        await self.agent.close()
        await self.multi_probe.close()

    def _prepare_user_memory(self, dataset: MemoryDataset, user_id: int):
        """Reset target memory and inject the target user's member memories."""
        if self._prepared_user_id == user_id:
            return

        user_set = dataset.get_user(user_id)
        injected = self.agent.prepare_user_memory(user_set)

        self._prepared_user_id = user_id
        self._prepared_memory_count = injected
        log.info(
            "Prepared %s target for user %s with %d member units",
            self.target,
            user_id,
            injected,
        )

    async def attack_user(
        self,
        dataset: MemoryDataset,
        user_attack_set: UserAttackSet,
        access_level: str,
    ) -> list[MembershipPrediction]:
        """Run the default direct multi-probe attack for a single user."""
        user_id = user_attack_set.user_id
        log.info("=" * 50)
        log.info("Attacking User %d", user_id)
        log.info("=" * 50)

        try:
            self._prepare_user_memory(dataset, user_id)
        except Exception as exc:
            log.error("Memory injection failed for user %d: %s", user_id, exc, exc_info=True)
            return [
                failed_prediction_for_unit(unit, "memory_injection", exc)
                for unit in user_attack_set.all_units
            ]

        return await self.multi_probe.attack(
            self.agent,
            user_attack_set.all_units,
            access_level,
            self.target,
        )

    async def attack_all_users(
        self,
        dataset: MemoryDataset,
        user_attack_sets: list[UserAttackSet],
        access_level: str,
    ) -> tuple[list[MembershipPrediction], dict[int, list[MembershipPrediction]]]:
        """Run the attack over all sampled users."""
        all_predictions: list[MembershipPrediction] = []
        per_user_predictions: dict[int, list[MembershipPrediction]] = {}

        for user_idx, user_attack_set in enumerate(user_attack_sets, start=1):
            log.info(
                "\n[User %d/%d] Processing user_id=%d",
                user_idx,
                len(user_attack_sets),
                user_attack_set.user_id,
            )
            try:
                user_preds = await self.attack_user(dataset, user_attack_set, access_level)
            except Exception as exc:
                log.error(
                    "User-level attack failed for user %d: %s",
                    user_attack_set.user_id,
                    exc,
                    exc_info=True,
                )
                user_preds = [
                    failed_prediction_for_unit(unit, "user_attack", exc)
                    for unit in user_attack_set.all_units
                ]
            per_user_predictions[user_attack_set.user_id] = user_preds
            all_predictions.extend(user_preds)

        return all_predictions, per_user_predictions

    async def evaluate(
        self,
        predictions: list[MembershipPrediction],
        access_level: str,
        seed: int,
        output_dir: Path,
        runtime_seconds: Optional[float] = None,
        save_probe_responses: bool = False,
    ) -> dict:
        al = AccessLevel(access_level)
        report = self.evaluator.full_report(predictions, al, seed)
        report["attack_method"] = "multi_probe_direct_no_contrastive"
        if runtime_seconds is not None:
            report["runtime_seconds"] = float(runtime_seconds)
            report["runtime_scope"] = "attack execution through prediction generation; excludes metrics/report serialization"
        report["direct_probe_k"] = self.direct_probe_k
        report["probe_generation"] = "single LLM call generating k direct-recall probes per memory"
        report["probe_type_ablation"] = compute_probe_type_ablation(
            predictions,
            al,
            seed,
            score_threshold=0.5,
        )
        report["top_n_response_ablation"] = compute_top_n_response_ablation(
            predictions,
            al,
            seed,
            score_threshold=0.5,
            max_n=self.direct_probe_k,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        report["probe_responses_saved"] = bool(save_probe_responses)
        self.evaluator.save_report(report, output_dir / "report.json")
        self.evaluator.save_predictions(
            predictions,
            output_dir / "predictions.json",
            include_probe_responses=save_probe_responses,
        )
        return report


async def amain():
    parser = argparse.ArgumentParser(
        description="CEA-MI natural attack over MemoryUnit datasets (default: direct multi-probe)"
    )
    parser.add_argument(
        "--target",
        choices=TARGET_CHOICES,
        default="nanobot",
        help="Memory-agent target to evaluate",
    )
    parser.add_argument(
        "--access",
        choices=["blackbox", "graybox", "whitebox"],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--num-facts",
        type=parse_optional_num_facts,
        default=None,
        help=(
            "attack units per class per user; use None/all/max or omit to sample "
            "min(members, non_members) for each user"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name or path (perltqa, test, msc, locomo, etc.)",
    )
    parser.add_argument("--db", default=None, help="Nanobot memory DB path (nanobot target only)")
    parser.add_argument(
        "--memory-file",
        default=None,
        help=(
            "Working memory JSON path for mem0/memgpt natural attacks. "
            "It is rewritten as each user's member memories are prepared."
        ),
    )
    parser.add_argument("--max-users", type=int, default=None, help="Optionally limit to the first N users")
    parser.add_argument("--output-path", default=None, help="Manually set the result output directory")
    parser.add_argument("--concurrency", type=int, default=40, help="Max concurrent LLM calls")
    parser.add_argument(
        "--direct-probe-k",
        type=int,
        default=5,
        help="Number of direct recall probes generated per memory",
    )
    parser.add_argument(
        "--response-scorer",
        choices=["rules", "llm"],
        default="rules",
        help="Response scoring backend: rules uses local matching, llm uses an LLM judge",
    )
    parser.add_argument(
        "--save-probe-responses",
        action="store_true",
        help="Include each probe's agent response in predictions.json (default: off)",
    )
    args = parser.parse_args()
    if args.access:
        log.warning(
            "--access is deprecated and ignored. natural_attack now runs whitebox once "
            "and writes blackbox/graybox/whitebox derived outputs."
        )

    cfg = Config()
    try:
        cfg.require_llm_config()
    except ValueError as exc:
        parser.error(str(exc))

    if args.db:
        cfg.nanobot_db_path = Path(args.db)
    if args.target == "nanobot" and args.memory_file:
        log.warning("--memory-file is ignored for nanobot target")
    if args.target != "nanobot" and args.db:
        log.warning("--db is ignored for %s target", args.target)

    dataset_name = args.dataset
    try:
        dataset_path = resolve_memory_dataset_path(args.dataset, cfg)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    dataset = load_memory_dataset(dataset_path)
    seeds = [args.seed]

    log.info("=" * 60)
    log.info("Target: %s", args.target)
    log.info("Dataset: %s", dataset_name)
    log.info("Loaded MemoryDataset from %s", dataset_path)
    log.info("Dataset stats: %s", dataset.stats())
    log.info("Using up to %s users", args.max_users if args.max_users is not None else "all")
    log.info("Default algorithm: multi_probe_direct_no_contrastive (k=%d)", args.direct_probe_k)
    log.info("=" * 60)

    all_reports = []
    for seed in seeds:
        cfg.seed = seed
        rng = random.Random(seed)

        user_attack_sets = sample_attack_units_per_user(
            dataset,
            num_per_class=args.num_facts,
            rng=rng,
            max_users=args.max_users,
        )

        total_members = sum(len(uas.members) for uas in user_attack_sets)
        total_nonmembers = sum(len(uas.non_members) for uas in user_attack_sets)
        num_facts_label = (
            "min(members, non_members)"
            if args.num_facts is None
            else str(args.num_facts)
        )

        run_access = "whitebox"
        output_name = (
            f"{dataset_name}_{args.target}_{args.response_scorer}"
            f"_k{args.direct_probe_k}_seed{seed}"
        )
        if args.output_path:
            output_dir = Path(args.output_path) / output_name
        else:
            output_dir = cfg.output_dir / output_name

        isolated_db: Optional[Path] = None
        target_memory_file: Optional[Path] = None

        if args.target == "nanobot":
            isolated_db = create_empty_memory_db(
                output_dir=cfg.output_dir,
                seed=seed,
                access_level=run_access,
                algo_name=f"multi_probe_{args.response_scorer}_{args.target}",
            )
            cfg.nanobot_db_path = isolated_db
            log.info("Created isolated DB: %s", isolated_db)
        else:
            target_memory_file = (
                Path(args.memory_file).expanduser()
                if args.memory_file
                else output_dir / f"{args.target}_working_memories.json"
            )
            target_memory_file.parent.mkdir(parents=True, exist_ok=True)
            log.info("Using %s working memory file: %s", args.target, target_memory_file)

        log.info("=" * 60)
        log.info("CEA-MI Natural Memory Attack (multi-probe direct)")
        log.info(
            "Target: %s | Run access: %s | Derived access: %s | Units/class/user: %s | Seed: %d | Concurrency: %d | Scorer: %s | k: %d",
            args.target,
            run_access,
            ",".join(ACCESS_DERIVATION_ORDER),
            num_facts_label,
            seed,
            args.concurrency,
            args.response_scorer,
            args.direct_probe_k,
        )
        if args.target == "nanobot":
            log.info("DB: %s", cfg.nanobot_db_path)
        else:
            log.info("Memory file: %s", target_memory_file)
        log.info(
            "Attack set: %d users, %d members + %d nonmembers = %d total",
            len(user_attack_sets),
            total_members,
            total_nonmembers,
            total_members + total_nonmembers,
        )
        log.info("=" * 60)

        attacker = NaturalAttack(
            cfg,
            rng,
            concurrency=args.concurrency,
            response_scorer=args.response_scorer,
            target=args.target,
            memory_file=target_memory_file,
            direct_probe_k=args.direct_probe_k,
        )
        start = time.time()
        try:
            all_predictions, per_user_predictions = await attacker.attack_all_users(
                dataset,
                user_attack_sets,
                run_access,
            )

            if not all_predictions:
                raise RuntimeError("No predictions generated across all users")

            elapsed = time.time() - start
            log.info("=" * 60)
            log.info("RESULTS (seed=%d)", seed)
            log.info("=" * 60)

            for access_level in ACCESS_DERIVATION_ORDER:
                al = AccessLevel(access_level)
                access_output_dir = output_dir / access_level
                projected_predictions = project_predictions_for_access(
                    all_predictions,
                    access_level,
                    score_threshold=0.5,
                )
                projected_per_user_predictions = project_per_user_predictions_for_access(
                    per_user_predictions,
                    access_level,
                    score_threshold=0.5,
                )
                global_report = await attacker.evaluate(
                    projected_predictions,
                    access_level,
                    seed,
                    access_output_dir,
                    runtime_seconds=elapsed,
                    save_probe_responses=args.save_probe_responses,
                )

                per_user_metrics = compute_per_user_metrics(
                    projected_per_user_predictions,
                    attacker.evaluator,
                    al,
                )

                log.info("--- %s global (all users combined) ---", access_level)
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
                log.info(
                    "Per-user ROC-AUC: mean=%.4f std=%.4f",
                    per_user_metrics.get("roc_auc_mean", 0.0),
                    per_user_metrics.get("roc_auc_std", 0.0),
                )
                log.info("Output:   %s", access_output_dir)

                meta = {
                    "version": "per_user_memoryunit_attack",
                    "attack_method": "multi_probe_direct_no_contrastive",
                    "target": args.target,
                    "access_level": access_level,
                    "derived_from_access": run_access,
                    "api_base": cfg.api_base,
                    "model": cfg.model,
                    "dataset_name": dataset_name,
                    "dataset_path": str(dataset_path),
                    "dataset_stats": dataset.stats(),
                    "nanobot_db_path": str(cfg.nanobot_db_path) if args.target == "nanobot" else None,
                    "memory_file": str(target_memory_file) if target_memory_file else None,
                    "num_users": len(user_attack_sets),
                    "total_members": total_members,
                    "total_nonmembers": total_nonmembers,
                    "runtime_seconds": elapsed,
                    "response_scorer": args.response_scorer,
                    "direct_probe_k": args.direct_probe_k,
                    "probe_responses_saved": args.save_probe_responses,
                    "attack_protocol": "run whitebox once, then derive blackbox/graybox/whitebox scores from saved evidence components",
                    "notes": [
                        "member/non-member labels come directly from MemoryDataset splits",
                        "each user's members are injected before testing that user's samples",
                        "non-members are never injected into agent memory",
                        "the default natural attack does not generate or score decoys",
                        "per_user_metrics shows metrics averaged across users",
                    ],
                }
                access_output_dir.mkdir(parents=True, exist_ok=True)
                with open(access_output_dir / "meta.json", "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2, ensure_ascii=False)

                with open(access_output_dir / "per_user_metrics.json", "w", encoding="utf-8") as f:
                    json.dump(per_user_metrics, f, indent=2, default=str)

                global_report["seed"] = seed
                global_report["per_user_metrics"] = per_user_metrics
                global_report["derived_from_access"] = run_access
                attacker.evaluator.save_report(global_report, access_output_dir / "report.json")
                roc_summary = global_report.get("roc_auc", {})
                roc_value = (
                    roc_summary.get("value", 0.0)
                    if isinstance(roc_summary, dict)
                    else roc_summary
                )
                comparison_payload = {
                    "target": args.target,
                    "access_level": access_level,
                    "run_access_level": run_access,
                    "dataset_name": dataset_name,
                    "dataset_path": str(dataset_path),
                    "seed": seed,
                    "response_scorer": args.response_scorer,
                    "direct_probe_k": args.direct_probe_k,
                    "results": {
                        "mrmmia": {
                            "roc_auc": roc_value,
                            "accuracy": global_report.get("accuracy", 0.0),
                            "runtime": elapsed,
                            "queries": global_report.get("total_queries", 0),
                            "run_access_level": run_access,
                            "direct_probe_k": args.direct_probe_k,
                        }
                    },
                }
                with open(access_output_dir / "comparison.json", "w", encoding="utf-8") as f:
                    json.dump(comparison_payload, f, indent=2)
                all_reports.append(global_report)
                print(json.dumps(global_report, indent=2, default=str))

            if isolated_db is not None and cleanup_isolated_db(isolated_db):
                log.info("Cleaned up isolated DB: %s", isolated_db)

        finally:
            await attacker.cleanup()
            if target_memory_file is not None and target_memory_file.exists():
                try:
                    target_memory_file.unlink()
                    log.info("Removed working memory file: %s", target_memory_file)
                except OSError as exc:
                    log.warning("Could not remove working memory file %s: %s", target_memory_file, exc)

    if len(all_reports) > 1:
        import numpy as np

        log.info("=" * 60)
        log.info("AGGREGATED RESULTS (%d seeds)", len(all_reports))
        for metric in ["accuracy", "brier_score", "avg_rounds_used"]:
            vals = [r.get(metric, 0.0) for r in all_reports]
            log.info("%s: mean=%.4f std=%.4f", metric, float(np.mean(vals)), float(np.std(vals)))


if __name__ == "__main__":
    asyncio.run(amain())

"""Shared helpers for MemoryDataset-based membership attacks."""
from __future__ import annotations

import argparse
import logging
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

from config import Config
from evaluation import Evaluator
from memory_extractor import MemoryExtractor
from memory_unit import MemoryDataset, MemoryUnit
from models import AccessLevel, Fact, MembershipPrediction, ProbeResult, RoundEvidence

log = logging.getLogger(__name__)


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


def failed_prediction_for_unit(unit: MemoryUnit, stage: str, reason: object) -> MembershipPrediction:
    """Count a sampled unit as a failed attack sample instead of dropping it."""
    return MembershipPrediction(
        fact_id=unit.id,
        is_member_true=unit.is_member,
        score=0.0,
        is_member_pred=False,
        failed_stage=stage,
        failure_reason=str(reason)[:500],
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


def resolve_memory_dataset_path(dataset_arg: str, cfg: Config) -> Path:
    """Resolve a MemoryDataset path from a file path or dataset alias."""
    candidate = Path(dataset_arg)
    if candidate.exists():
        return candidate

    if dataset_arg.strip().lower() == "test":
        test_path = cfg.data_dir / "test_perltqa_dialogue_seed42.json"
        if test_path.exists():
            return test_path

    candidates = [
        cfg.data_dir / f"{dataset_arg}_seed42.json",
        cfg.data_dir / f"{dataset_arg}_dialogue_seed42.json",
    ]
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Dataset not found: "
        f"{dataset_arg} (also tried {', '.join(str(path) for path in candidates)})"
    )


@dataclass
class UserAttackSet:
    """Attack samples for a single user."""

    user_id: int
    members: list[MemoryUnit]
    non_members: list[MemoryUnit]

    @property
    def all_units(self) -> list[MemoryUnit]:
        return self.members + self.non_members


def sample_attack_units_per_user(
    dataset: MemoryDataset,
    num_per_class: Optional[int],
    rng: random.Random,
    max_users: Optional[int] = None,
) -> list[UserAttackSet]:
    """Sample a balanced attack set for each user."""
    user_ids = dataset.user_ids
    if max_users is not None:
        user_ids = user_ids[:max_users]

    attack_sets: list[UserAttackSet] = []
    for user_id in user_ids:
        user_set = dataset.get_user(user_id)
        available_per_class = min(len(user_set.members), len(user_set.non_members))
        n_samples = (
            available_per_class
            if num_per_class is None
            else min(num_per_class, available_per_class)
        )

        if n_samples == 0:
            log.warning(
                "User %d has insufficient data (members=%d, non_members=%d), skipping",
                user_id,
                len(user_set.members),
                len(user_set.non_members),
            )
            continue

        attack_sets.append(
            UserAttackSet(
                user_id=user_id,
                members=rng.sample(user_set.members, n_samples),
                non_members=rng.sample(user_set.non_members, n_samples),
            )
        )

        log.info(
            "User %d: sampled %d members, %d non-members (from %d/%d available)",
            user_id,
            n_samples,
            n_samples,
            len(user_set.members),
            len(user_set.non_members),
        )

    if not attack_sets:
        raise ValueError("No users with sufficient data for attack")

    return attack_sets


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

        user_metrics[user_id] = evaluator.full_report(preds, access_level, seed=0)

    if not user_metrics:
        return {"error": "No users with valid metrics"}

    agg: dict[str, object] = {}
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


PROBE_TYPE_ORDER = (
    "direct_recall",
    "indirect_reasoning",
    "provenance",
    "confirmation",
)


def _round_probe_type(evidence: RoundEvidence) -> str:
    return (
        evidence.probe_type.value
        if hasattr(evidence.probe_type, "value")
        else str(evidence.probe_type)
    )


def _prediction_from_round_subset(
    pred: MembershipPrediction,
    selected_types: set[str],
    score_threshold: float,
) -> MembershipPrediction:
    selected = [e for e in pred.evidence_trail if _round_probe_type(e) in selected_types]
    score = sum(e.delta_score for e in selected) / len(selected) if selected else 0.0
    return MembershipPrediction(
        fact_id=pred.fact_id,
        is_member_true=pred.is_member_true,
        score=score,
        is_member_pred=score > score_threshold,
        evidence_trail=selected,
        num_rounds_used=len(selected),
        failed_stage=pred.failed_stage,
        failure_reason=pred.failure_reason,
    )


def _probe_subset_report(
    predictions: list[MembershipPrediction],
    selected_types: set[str],
    access_level: AccessLevel,
    seed: int,
    score_threshold: float,
    bootstrap_n: int,
) -> dict:
    subset_predictions = [
        _prediction_from_round_subset(pred, selected_types, score_threshold)
        for pred in predictions
    ]
    report = Evaluator(bootstrap_n=bootstrap_n, seed=seed).full_report(
        subset_predictions,
        access_level,
        seed,
    )
    report["probe_types_used"] = sorted(selected_types)
    report["aggregation"] = "mean(delta_score over selected probe rounds); missing selected rounds score as 0"
    report["score_threshold"] = score_threshold
    report["bootstrap_n"] = bootstrap_n
    return report


def compute_probe_type_ablation(
    predictions: list[MembershipPrediction],
    access_level: AccessLevel,
    seed: int,
    score_threshold: float = 0.0,
    bootstrap_n: int = 0,
) -> dict:
    observed = []
    for pred in predictions:
        for evidence in pred.evidence_trail:
            probe_type = _round_probe_type(evidence)
            if probe_type not in observed:
                observed.append(probe_type)

    ordered_types = [t for t in PROBE_TYPE_ORDER if t in observed]
    ordered_types.extend(t for t in observed if t not in ordered_types)
    if not ordered_types:
        return {
            "available": False,
            "reason": "no evidence_trail with probe_type was found",
        }

    single = {
        probe_type: _probe_subset_report(
            predictions,
            {probe_type},
            access_level,
            seed,
            score_threshold,
            bootstrap_n,
        )
        for probe_type in ordered_types
    }
    leave_one_out = {}
    if len(ordered_types) > 1:
        for probe_type in ordered_types:
            selected = set(ordered_types) - {probe_type}
            leave_one_out[f"without_{probe_type}"] = _probe_subset_report(
                predictions,
                selected,
                access_level,
                seed,
                score_threshold,
                bootstrap_n,
            )

    return {
        "available": True,
        "probe_types": ordered_types,
        "single_probe": single,
        "leave_one_out": leave_one_out,
    }


def _prediction_from_top_n_responses(
    pred: MembershipPrediction,
    n: int,
    score_threshold: float,
) -> MembershipPrediction:
    selected = sorted(pred.evidence_trail, key=lambda evidence: evidence.round_idx)[:n]
    score = sum(e.delta_score for e in selected) / len(selected) if selected else 0.0
    return MembershipPrediction(
        fact_id=pred.fact_id,
        is_member_true=pred.is_member_true,
        score=score,
        is_member_pred=score > score_threshold,
        evidence_trail=selected,
        num_rounds_used=len(selected),
        failed_stage=pred.failed_stage,
        failure_reason=pred.failure_reason,
    )


def _top_n_response_report(
    predictions: list[MembershipPrediction],
    n: int,
    access_level: AccessLevel,
    seed: int,
    score_threshold: float,
    bootstrap_n: int,
) -> dict:
    subset_predictions = [
        _prediction_from_top_n_responses(pred, n, score_threshold)
        for pred in predictions
    ]
    report = Evaluator(bootstrap_n=bootstrap_n, seed=seed).full_report(
        subset_predictions,
        access_level,
        seed,
    )
    report["top_n"] = n
    report["aggregation"] = "mean(first-n delta_score); missing selected rounds score as 0"
    report["score_threshold"] = score_threshold
    report["bootstrap_n"] = bootstrap_n
    return report


def compute_top_n_response_ablation(
    predictions: list[MembershipPrediction],
    access_level: AccessLevel,
    seed: int,
    score_threshold: float = 0.5,
    max_n: Optional[int] = None,
    bootstrap_n: int = 0,
) -> dict:
    max_observed = max((len(pred.evidence_trail) for pred in predictions), default=0)
    if max_observed == 0:
        return {
            "available": False,
            "reason": "no evidence_trail was found",
        }

    limit = max_observed if max_n is None else min(max_observed, max(1, max_n))
    return {
        "available": True,
        "selection": "per sample: use the first n generated probe responses in generation order",
        "max_observed_rounds": max_observed,
        "top_n": {
            f"top_{n}": _top_n_response_report(
                predictions,
                n,
                access_level,
                seed,
                score_threshold,
                bootstrap_n,
            )
            for n in range(1, limit + 1)
        },
    }


ACCESS_DERIVATION_ORDER = ("blackbox", "graybox", "whitebox")
ACCESS_FEATURE_KEYS = {
    "blackbox": {"delta_response_score"},
    "graybox": {"delta_response_score", "delta_logprob"},
    "whitebox": {"delta_response_score", "delta_logprob", "delta_memory_statement_score"},
}
ACCESS_FEATURE_WEIGHTS = {
    "delta_response_score": 1.0,
    "delta_memory_statement_score": 1.0,
    "delta_logprob": 1.0,
}
MEMORY_FEATURE_KEYS = {
    "memory_statement_scorer",
    "fact_memory_statement_scores",
    "decoy_memory_statement_scores",
    "fact_memory_statement_score_max",
    "decoy_memory_statement_score_max",
    "fact_memory_statement_score_mean",
    "decoy_memory_statement_score_mean",
    "delta_memory_statement_score",
    "delta_recall_rate",
    "delta_recall_similarity",
    "fact_memory_candidate_statement",
    "fact_memory_key_value",
}
LOGPROB_FEATURE_KEYS = {"delta_logprob", "raw_delta_logprob"}


def _derived_round_score(features: dict, access_level: str) -> float:
    allowed = ACCESS_FEATURE_KEYS[access_level]
    total = 0.0
    total_weight = 0.0
    for key, weight in ACCESS_FEATURE_WEIGHTS.items():
        if key in allowed and key in features:
            total += weight * features[key]
            total_weight += weight
    return total / total_weight if total_weight > 0 else 0.0


def _project_features_for_access(features: dict, access_level: str) -> dict:
    projected = dict(features)
    if access_level == "blackbox":
        for key in LOGPROB_FEATURE_KEYS | MEMORY_FEATURE_KEYS:
            projected.pop(key, None)
    elif access_level == "graybox":
        for key in MEMORY_FEATURE_KEYS:
            projected.pop(key, None)
    elif access_level != "whitebox":
        raise ValueError(f"Unknown access level: {access_level}")
    return projected


def _project_probe_result_for_access(result: ProbeResult, access_level: str) -> ProbeResult:
    projected = replace(result)
    if access_level == "blackbox":
        projected.logprobs = None
        projected.mean_logprob = None
    if access_level != "whitebox":
        projected.recall_triggered = None
        projected.recall_top_similarity = None
        projected.recall_hit_count = None
        projected.memory_metadata = None
    return projected


def project_prediction_for_access(
    prediction: MembershipPrediction,
    access_level: str,
    score_threshold: float = 0.5,
) -> MembershipPrediction:
    """Project one whitebox/graybox prediction to an access-specific score."""
    if access_level not in ACCESS_FEATURE_KEYS:
        raise ValueError(f"Unknown access level: {access_level}")

    projected_evidence: list[RoundEvidence] = []
    for evidence in prediction.evidence_trail:
        features = _project_features_for_access(evidence.features, access_level)
        round_score = _derived_round_score(features, access_level)
        projected_evidence.append(
            replace(
                evidence,
                delta_score=round_score,
                features=features,
                fact_results=[
                    _project_probe_result_for_access(result, access_level)
                    for result in evidence.fact_results
                ],
                decoy_results=[
                    _project_probe_result_for_access(result, access_level)
                    for result in evidence.decoy_results
                ],
            )
        )

    score = (
        sum(evidence.delta_score for evidence in projected_evidence) / len(projected_evidence)
        if projected_evidence
        else prediction.score
    )
    return replace(
        prediction,
        score=score,
        is_member_pred=score > score_threshold,
        evidence_trail=projected_evidence,
        num_rounds_used=len(projected_evidence) if projected_evidence else prediction.num_rounds_used,
    )


def project_predictions_for_access(
    predictions: list[MembershipPrediction],
    access_level: str,
    score_threshold: float = 0.5,
) -> list[MembershipPrediction]:
    return [
        project_prediction_for_access(prediction, access_level, score_threshold)
        for prediction in predictions
    ]


def project_per_user_predictions_for_access(
    per_user_predictions: dict[int, list[MembershipPrediction]],
    access_level: str,
    score_threshold: float = 0.5,
) -> dict[int, list[MembershipPrediction]]:
    return {
        user_id: project_predictions_for_access(predictions, access_level, score_threshold)
        for user_id, predictions in per_user_predictions.items()
    }


def parse_optional_num_facts(value: str) -> Optional[int]:
    normalized = value.strip().lower()
    if normalized in {"none", "all", "max"}:
        return None
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("--num-facts must be non-negative, None, all, or max")
    return parsed

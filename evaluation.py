"""Evaluation -- metrics, calibration, ablation, and reporting for CEA-MI."""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
from models import MembershipPrediction, ExperimentResult, AccessLevel


TPR_FPR_TARGETS = (0.1, 0.01, 0.005, 1e-3, 1e-4)


class Evaluator:
    def __init__(self, bootstrap_n: int = 1000, seed: int = 42):
        self.bootstrap_n = bootstrap_n
        self.rng = np.random.RandomState(seed)

    @staticmethod
    def _tpr_at_fpr(fpr_arr, tpr_arr, fpr_target):
        valid = fpr_arr <= fpr_target
        return float(tpr_arr[valid][-1]) if valid.any() else 0.0

    @staticmethod
    def _ece(y_true, y_prob, n_bins=10):
        try:
            pt, pp = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
            bc = np.histogram(y_prob, bins=n_bins, range=(0,1))[0]
            total = y_true.shape[0]
            return float(sum((bc[i]/total)*abs(pt[i]-pp[i]) for i in range(len(pt)) if i < len(bc) and bc[i] > 0))
        except Exception:
            return 0.0

    @staticmethod
    def _to_prob(scores):
        """Map raw scores to [0,1] via sigmoid for metrics that need probabilities."""
        # Scale factor: 5.0 maps score range ~[-1,1] to sigmoid range ~[0.007, 0.993]
        return 1.0 / (1.0 + np.exp(-5.0 * scores))

    @staticmethod
    def _prediction_prob(prediction: MembershipPrediction) -> float:
        return float(Evaluator._to_prob(np.array([prediction.score]))[0])

    @staticmethod
    def _prediction_label(prediction: MembershipPrediction) -> int:
        return int(prediction.is_member_pred)

    def evaluate(self, predictions, access_level=AccessLevel.BLACKBOX, seed=0):
        y_true = np.array([int(p.is_member_true) for p in predictions])
        y_score = np.array([p.score for p in predictions])
        y_prob = np.array([self._prediction_prob(p) for p in predictions])
        result = ExperimentResult(seed=seed, access_level=access_level, predictions=predictions)
        if len(np.unique(y_true)) < 2:
            return result
        # ROC-AUC and PR-AUC work with any score range (rank-based)
        result.roc_auc = float(roc_auc_score(y_true, y_score))
        result.pr_auc = float(average_precision_score(y_true, y_score))
        fpr_arr, tpr_arr, _ = roc_curve(y_true, y_score)
        for ft in TPR_FPR_TARGETS:
            result.tpr_at_fpr[f"fpr={ft}"] = self._tpr_at_fpr(fpr_arr, tpr_arr, ft)
        # Brier and ECE need probabilities in [0,1]
        result.brier_score = float(brier_score_loss(y_true, y_prob))
        result.ece = self._ece(y_true, y_prob)
        result.total_queries = sum(p.num_rounds_used for p in predictions)
        return result

    def bootstrap_ci(self, predictions, metric_fn, alpha=0.05):
        y_true = np.array([int(p.is_member_true) for p in predictions])
        y_score = np.array([p.score for p in predictions])
        point = metric_fn(y_true, y_score)
        boots = []
        n = len(predictions)
        for _ in range(self.bootstrap_n):
            idx = self.rng.choice(n, size=n, replace=True)
            yt, ys = y_true[idx], y_score[idx]
            if len(np.unique(yt)) < 2:
                continue
            try:
                boots.append(metric_fn(yt, ys))
            except Exception:
                continue
        if not boots:
            return float(point), float(point), float(point)
        return float(point), float(np.percentile(boots, 100*alpha/2)), float(np.percentile(boots, 100*(1-alpha/2)))

    def full_report(self, predictions, access_level=AccessLevel.BLACKBOX, seed=0):
        result = self.evaluate(predictions, access_level, seed)
        y_true = np.array([int(p.is_member_true) for p in predictions])
        y_score = np.array([p.score for p in predictions])
        report = {"seed": seed, "access_level": access_level.value,
                  "n_samples": len(predictions), "n_members": int(y_true.sum()),
                  "n_nonmembers": int((1-y_true).sum())}
        if len(np.unique(y_true)) < 2:
            report["error"] = "Not enough class diversity"
            return report
        auc_p, auc_lo, auc_hi = self.bootstrap_ci(predictions, lambda yt, ys: roc_auc_score(yt, ys))
        report["roc_auc"] = {"value": auc_p, "ci_lower": auc_lo, "ci_upper": auc_hi}
        pr_p, pr_lo, pr_hi = self.bootstrap_ci(predictions, lambda yt, ys: average_precision_score(yt, ys))
        report["pr_auc"] = {"value": pr_p, "ci_lower": pr_lo, "ci_upper": pr_hi}
        for ft in TPR_FPR_TARGETS:
            def tpr_fn(yt, ys, _ft=ft):
                f, t, _ = roc_curve(yt, ys); return self._tpr_at_fpr(f, t, _ft)
            tp, tl, th = self.bootstrap_ci(predictions, tpr_fn)
            report[f"tpr_at_fpr_{ft}"] = {"value": tp, "ci_lower": tl, "ci_upper": th}
        report["brier_score"] = result.brier_score
        report["ece"] = result.ece
        report["avg_rounds_used"] = float(np.mean([p.num_rounds_used for p in predictions]))
        report["total_queries"] = result.total_queries
        y_pred = np.array([self._prediction_label(p) for p in predictions])
        report["accuracy"] = float((y_pred == y_true).mean())
        report["generated_at"] = datetime.now(timezone.utc).isoformat()
        return report

    def save_report(self, report, output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, default=str))

    @staticmethod
    def _probe_type_value(value) -> str:
        return value.value if hasattr(value, "value") else str(value)

    @staticmethod
    def _serialize_probe_result(result, evidence, side: str, include_response: bool = False) -> dict:
        probe = result.probe
        topic = probe.topic or ""
        key_value = probe.expected_if_member or ""
        if side == "fact":
            topic = topic or evidence.features.get("probe_topic", "")
            key_value = evidence.features.get("probe_key_value", "") or key_value
        if Evaluator._probe_type_value(probe.probe_type) == "judge_yes_no":
            key_value = evidence.features.get("probe_key_value", "")
        record = {
            "round_idx": evidence.round_idx,
            "side": side,
            "probe_id": probe.id,
            "probe_type": Evaluator._probe_type_value(probe.probe_type),
            "topic": topic,
            "key_value": key_value,
            "question": probe.question,
            "expected_if_member": probe.expected_if_member,
            "expected_if_nonmember": probe.expected_if_nonmember,
            "perspective_idx": probe.perspective_idx,
        }
        if getattr(probe, "metadata", None):
            record["probe_metadata"] = probe.metadata
        if include_response:
            record["response"] = result.response
            record["latency_ms"] = result.latency_ms
            record["mean_logprob"] = result.mean_logprob
            if result.recall_triggered is not None:
                record["recall_triggered"] = result.recall_triggered
            if result.recall_top_similarity is not None:
                record["recall_top_similarity"] = result.recall_top_similarity
            if result.recall_hit_count is not None:
                record["recall_hit_count"] = result.recall_hit_count
            if result.memory_metadata is not None:
                record["memory_metadata"] = result.memory_metadata
        return record

    @staticmethod
    def _serialize_component_scores(features: dict) -> dict:
        """Raw, unweighted score components used to derive each round score."""
        keys = [
            "delta_response_score",
            "delta_logprob",
            "raw_delta_logprob",
            "delta_memory_statement_score",
            "fact_response_score_mean",
            "decoy_response_score_mean",
            "fact_memory_statement_score_max",
            "decoy_memory_statement_score_max",
            "fact_memory_statement_score_mean",
            "decoy_memory_statement_score_mean",
        ]
        return {key: features[key] for key in keys if key in features}

    def prediction_rows(self, predictions, include_probe_responses: bool = False):
        rows = []
        for p in predictions:
            row = {
                "fact_id": p.fact_id,
                "is_member_true": p.is_member_true,
                "score": p.score,
                "is_member_pred": p.is_member_pred,
                "num_rounds": p.num_rounds_used,
            }
            if p.failed_stage:
                row["failed_stage"] = p.failed_stage
            if p.failure_reason:
                row["failure_reason"] = p.failure_reason
            if p.evidence_trail:
                probe_records = []
                for evidence in p.evidence_trail:
                    probe_records.extend(
                        self._serialize_probe_result(
                            result,
                            evidence,
                            "fact",
                            include_response=include_probe_responses,
                        )
                        for result in evidence.fact_results
                    )
                    probe_records.extend(
                        self._serialize_probe_result(
                            result,
                            evidence,
                            "decoy",
                            include_response=include_probe_responses,
                        )
                        for result in evidence.decoy_results
                    )
                if probe_records:
                    row["probes"] = probe_records
                    row["probe_topics"] = [record["topic"] for record in probe_records]
                    row["probe_key_values"] = [record["key_value"] for record in probe_records]

                row["round_scores"] = [
                    {
                        "round_idx": evidence.round_idx,
                        "probe_type": self._probe_type_value(evidence.probe_type),
                        "delta_score": evidence.delta_score,
                        "score_fact": evidence.score_fact,
                        "score_decoy": evidence.score_decoy,
                        "component_scores": self._serialize_component_scores(evidence.features),
                        "features": evidence.features,
                        "topic": evidence.features.get("probe_topic")
                        or (
                            evidence.fact_results[0].probe.topic
                            if evidence.fact_results
                            else ""
                        ),
                        "key_value": evidence.features.get("probe_key_value")
                        or (
                            evidence.fact_results[0].probe.expected_if_member
                            if evidence.fact_results
                            else ""
                        ),
                        "question": (
                            evidence.fact_results[0].probe.question
                            if evidence.fact_results
                            else ""
                        ),
                    }
                    for evidence in p.evidence_trail
                ]
            rows.append(row)
        return rows

    def save_predictions(self, predictions, output_path: Path, include_probe_responses: bool = False):
        rows = self.prediction_rows(
            predictions,
            include_probe_responses=include_probe_responses,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(rows, indent=2))

"""Feature extractor -- extract membership signals from probe results."""
from __future__ import annotations
from models import Fact, Probe, ProbeResult, ProbeType, RoundEvidence

HEDGING_WORDS = {
    "maybe", "perhaps", "possibly", "might", "could", "uncertain",
    "not sure", "i think", "i believe", "it seems", "probably",
    "likely", "unlikely", "not certain", "unclear",
    "no record", "no information",
}

class FeatureExtractor:
    def __init__(self):
        self.learned_weights = None  # set after calibration

    @staticmethod
    def text_similarity(response: str, expected: str) -> float:
        if not expected or not response:
            return 0.0
        rl, el = response.lower(), expected.lower()
        if el in rl:
            return 1.0
        rt, et = set(rl.split()), set(el.split())
        return len(rt & et) / len(et) if et else 0.0

    @staticmethod
    def hedging_ratio(response: str) -> float:
        if not response:
            return 0.0
        rl = response.lower()
        wc = len(rl.split())
        return sum(1 for h in HEDGING_WORDS if h in rl) / max(wc, 1) if wc else 0.0

    @staticmethod
    def response_confidence(response: str) -> float:
        rl = response.lower()
        for sig in ["i don't have", "no information", "i cannot", "i don't know"]:
            if sig in rl:
                return 0.1
        hc = sum(1 for h in HEDGING_WORDS if h in rl)
        cc = sum(1 for c in ["yes", "your", "you mentioned", "i remember", "based on"] if c in rl)
        return max(0.0, min(1.0, 0.5 + 0.1 * cc - 0.1 * hc))

    @staticmethod
    def consistency_score(results: list[ProbeResult]) -> float:
        responses = [r.response.lower().strip() for r in results if r.response]
        if len(responses) < 2:
            return 0.5
        sims = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                ti, tj = set(responses[i].split()), set(responses[j].split())
                if ti and tj:
                    sims.append(2 * len(ti & tj) / (len(ti) + len(tj)))
        return sum(sims) / len(sims) if sims else 0.5

    @staticmethod
    def contradiction_stability(response: str, expected_value: str) -> float:
        if not expected_value:
            return 0.0
        if expected_value.lower() in response.lower():
            boost = sum(0.1 for s in ["actually", "correct", "in fact", "remember"] if s in response.lower())
            return min(1.0, 0.7 + boost)
        return 0.2

    @staticmethod
    def confirmation_score(response: str, expected_value: str) -> float:
        """Score how strongly the model confirms a specific value.

        Returns high score when the model confirms ("yes", "correct", "that's right")
        and low score when it denies or is uncertain.
        """
        if not expected_value or not response:
            return 0.0
        rl = response.lower()
        # Check if the value appears in response
        value_present = expected_value.lower() in rl
        # Check for confirmation signals
        confirm_signals = ["yes", "correct", "that's right", "right", "exactly",
                           "indeed", "as you mentioned", "i remember", "you did mention",
                           "you told me", "you said", "you shared"]
        deny_signals = ["no", "i don't", "i'm not sure", "i don't have",
                        "i don't recall", "i don't remember", "not aware",
                        "no record", "haven't mentioned", "didn't mention"]
        confirm_count = sum(1 for s in confirm_signals if s in rl)
        deny_count = sum(1 for s in deny_signals if s in rl)
        if value_present and confirm_count > 0:
            return min(1.0, 0.7 + 0.1 * confirm_count)
        elif value_present:
            return 0.5
        elif deny_count > 0:
            return max(0.0, 0.2 - 0.05 * deny_count)
        return 0.3

    def extract_round_features(self, fact: Fact, fact_results: list[ProbeResult],
                               decoy_results: list[ProbeResult], probe_type: ProbeType) -> dict:
        f = {}
        # P0: Contrastive correctness
        fs = [self.text_similarity(r.response, fact.key_value) for r in fact_results]
        ds = [self.text_similarity(r.response, r.probe.expected_if_member) for r in decoy_results]
        f["fact_similarity_mean"] = sum(fs) / len(fs) if fs else 0.0
        f["decoy_similarity_mean"] = sum(ds) / len(ds) if ds else 0.0
        f["delta_similarity"] = f["fact_similarity_mean"] - f["decoy_similarity_mean"]

        # P0: Cross-paraphrase consistency
        f["fact_consistency"] = self.consistency_score(fact_results)
        f["decoy_consistency"] = self.consistency_score(decoy_results)
        f["delta_consistency"] = f["fact_consistency"] - f["decoy_consistency"]

        # P1: Contradiction stability
        if probe_type == ProbeType.CONTRADICTION:
            fcs = [self.contradiction_stability(r.response, fact.key_value) for r in fact_results]
            dcs = [self.contradiction_stability(r.response, r.probe.expected_if_member) for r in decoy_results]
            f["delta_contradiction_stability"] = (sum(fcs)/len(fcs) if fcs else 0) - (sum(dcs)/len(dcs) if dcs else 0)

        # P1: Confirmation score (value-specific confirmation/denial)
        if probe_type == ProbeType.CONFIRMATION:
            fconf = [self.confirmation_score(r.response, fact.key_value) for r in fact_results]
            dconf = [self.confirmation_score(r.response, r.probe.expected_if_member) for r in decoy_results]
            f["delta_confirmation"] = (sum(fconf)/len(fconf) if fconf else 0) - (sum(dconf)/len(dconf) if dconf else 0)

        # P1: Retrieval strength (white-box)
        fr = [r for r in fact_results if r.recall_triggered is not None]
        dr = [r for r in decoy_results if r.recall_triggered is not None]
        if fr and dr:
            f["delta_recall_rate"] = (sum(1 for r in fr if r.recall_triggered)/len(fr)) - (sum(1 for r in dr if r.recall_triggered)/len(dr))
            f["delta_recall_similarity"] = max((r.recall_top_similarity or 0) for r in fr) - max((r.recall_top_similarity or 0) for r in dr)

        # P1: Logprob (gray-box) — normalize to ~[-1, 1]
        flp = [r.mean_logprob for r in fact_results if r.mean_logprob is not None]
        dlp = [r.mean_logprob for r in decoy_results if r.mean_logprob is not None]
        if flp and dlp:
            raw = sum(flp)/len(flp) - sum(dlp)/len(dlp)
            f["delta_logprob"] = max(-1.0, min(1.0, raw / 5.0))

        # P2: Latency — normalize ms to ~[-1, 1]
        flat = [r.latency_ms for r in fact_results if r.latency_ms > 0]
        dlat = [r.latency_ms for r in decoy_results if r.latency_ms > 0]
        if flat and dlat:
            raw = sum(flat)/len(flat) - sum(dlat)/len(dlat)
            f["delta_latency"] = max(-1.0, min(1.0, raw / 2000.0))

        # P2: Hedging (reversed: more hedging for decoy = member signal)
        fh = [self.hedging_ratio(r.response) for r in fact_results]
        dh = [self.hedging_ratio(r.response) for r in decoy_results]
        f["delta_hedging"] = (sum(dh)/len(dh) if dh else 0) - (sum(fh)/len(fh) if fh else 0)

        # Confidence
        fc = [self.response_confidence(r.response) for r in fact_results]
        dc = [self.response_confidence(r.response) for r in decoy_results]
        f["delta_confidence"] = (sum(fc)/len(fc) if fc else 0.5) - (sum(dc)/len(dc) if dc else 0.5)
        return f

    DEFAULT_WEIGHTS = {
        "delta_similarity": 3.0, "delta_consistency": 2.0,
        "delta_contradiction_stability": 2.0, "delta_confirmation": 3.5,
        "delta_recall_rate": 2.5, "delta_recall_similarity": 1.5,
        "delta_logprob": 1.5, "delta_confidence": 1.0,
        "delta_hedging": 0.8, "delta_latency": 0.3,
    }

    def learn_weights_from_calibration(self, calibration_data: list[tuple[dict, bool]]):
        """Learn feature weights using Fisher's Linear Discriminant.

        Args:
            calibration_data: list of (feature_dict, is_member) tuples
                collected during calibration probing.
        Returns:
            dict of learned weights (also stored on self.learned_weights)
        """
        from collections import defaultdict

        member_feats = defaultdict(list)
        nonmember_feats = defaultdict(list)

        for features, is_member in calibration_data:
            target = member_feats if is_member else nonmember_feats
            for k, v in features.items():
                if k.startswith("delta_"):
                    target[k].append(v)

        all_keys = set(list(member_feats.keys()) + list(nonmember_feats.keys()))
        weights = {}

        for key in all_keys:
            m_vals = member_feats.get(key, [0.0])
            nm_vals = nonmember_feats.get(key, [0.0])

            m_mean = sum(m_vals) / len(m_vals)
            nm_mean = sum(nm_vals) / len(nm_vals)

            m_var = sum((v - m_mean) ** 2 for v in m_vals) / max(len(m_vals) - 1, 1)
            nm_var = sum((v - nm_mean) ** 2 for v in nm_vals) / max(len(nm_vals) - 1, 1)

            pooled_std = ((m_var + nm_var) / 2) ** 0.5 + 1e-6
            fisher = abs(m_mean - nm_mean) / pooled_std

            # Anti-correlated features (nonmembers score higher) are unreliable
            if m_mean < nm_mean:
                fisher *= 0.1

            weights[key] = max(fisher, 0.1)

        self.learned_weights = weights
        return weights

    def compute_round_score(self, features: dict) -> float:
        weights = self.learned_weights if self.learned_weights else self.DEFAULT_WEIGHTS
        s, tw = 0.0, 0.0
        for k, w in weights.items():
            if k in features:
                s += w * features[k]
                tw += w
        return s / tw if tw > 0 else 0.0

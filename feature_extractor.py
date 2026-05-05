"""Feature extractor -- score probe responses for membership signals."""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from models import Fact, ProbeResult, ProbeType


UNKNOWN_PHRASES = (
    "i don't know",
    "i do not know",
    "don't know",
    "do not know",
    "i don't remember",
    "i do not remember",
    "don't remember",
    "do not remember",
    "i can't remember",
    "i cannot remember",
    "can't remember",
    "cannot remember",
    "i don't recall",
    "i do not recall",
    "don't recall",
    "do not recall",
    "i can't recall",
    "i cannot recall",
    "can't recall",
    "cannot recall",
    "no information",
    "no record",
    "not aware",
    "i don't have",
    "i do not have",
    "can't access",
    "cannot access",
)

HEDGING_PHRASES = (
    "maybe",
    "perhaps",
    "possibly",
    "might",
    "could be",
    "not sure",
    "i think",
    "i believe",
    "it seems",
    "probably",
    "likely",
    "uncertain",
    "unclear",
)

AFFIRMATION_PATTERNS = (
    r"\byes\b",
    r"\bcorrect\b",
    r"\bexactly\b",
    r"\bindeed\b",
    r"\btrue\b",
    r"\baccurate\b",
    r"\bthat's right\b",
    r"\bthat is right\b",
    r"\bthat's correct\b",
    r"\bthat is correct\b",
    r"\byou're right\b",
    r"\byou are right\b",
)

DENIAL_PATTERNS = (
    r"\bno\b",
    r"\bincorrect\b",
    r"\bwrong\b",
    r"\bnot correct\b",
    r"\bnot right\b",
    r"\bthat's not right\b",
    r"\bthat is not right\b",
    r"\bthat's not correct\b",
    r"\bthat is not correct\b",
    r"\bi don't think so\b",
    r"\bi do not think so\b",
    r"\bwasn't\b",
    r"\bwasnt\b",
    r"\bisn't\b",
    r"\bisnt\b",
    r"\baren't\b",
    r"\barent\b",
    r"\bweren't\b",
    r"\bwerent\b",
    r"\bdidn't\b",
    r"\bdidnt\b",
)

SOURCE_MEMORY_PHRASES = (
    "you told me",
    "you mentioned",
    "you said",
    "you shared",
    "you told me before",
    "you previously told me",
    "you previously mentioned",
    "i remember",
    "i recall",
    "from our previous conversation",
    "from what you told me",
    "from what you shared",
    "based on what you told me",
    "based on what you shared",
    "based on what you mentioned",
)

SOURCE_INFERRED_PHRASES = (
    "i infer",
    "i'd infer",
    "i would infer",
    "i guess",
    "i'd guess",
    "i would guess",
    "sounds like",
    "based on the question",
)

NEGATION_TOKENS = {
    "no",
    "not",
    "never",
    "without",
    "none",
    "neither",
    "don't",
    "dont",
    "doesn't",
    "doesnt",
    "didn't",
    "didnt",
    "isn't",
    "isnt",
    "aren't",
    "arent",
    "wasn't",
    "wasnt",
    "weren't",
    "werent",
    "can't",
    "cant",
    "cannot",
}


@dataclass(frozen=True)
class ResponseSignals:
    value_score: float
    value_present: bool
    value_negated: bool
    unknown: bool
    affirmed: bool
    denied: bool
    hedged: bool
    source_score: float


class FeatureExtractor:
    @staticmethod
    def _normalize(text: str) -> str:
        text = (text or "").lower()
        text = (
            text.replace("’", "'")
            .replace("‘", "'")
            .replace("“", '"')
            .replace("”", '"')
        )
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"[a-z0-9']+", FeatureExtractor._normalize(text))

    @staticmethod
    def _phrase_pattern(phrase: str) -> re.Pattern:
        normalized = FeatureExtractor._normalize(phrase)
        if not normalized:
            return re.compile(r"a^")
        parts = [re.escape(part) for part in normalized.split()]
        body = r"\s+".join(parts)
        return re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])", re.IGNORECASE)

    @staticmethod
    def _has_phrase(text: str, phrase: str) -> bool:
        return bool(FeatureExtractor._phrase_pattern(phrase).search(text))

    @staticmethod
    def _has_any_phrase(text: str, phrases: tuple[str, ...]) -> bool:
        return any(phrase in text for phrase in phrases)

    @staticmethod
    def _has_any_pattern(text: str, patterns: tuple[str, ...]) -> bool:
        return any(re.search(pattern, text) for pattern in patterns)

    @staticmethod
    def _token_overlap_score(response: str, expected: str) -> float:
        rt = set(FeatureExtractor._tokens(response))
        et = set(FeatureExtractor._tokens(expected))
        if not rt or not et:
            return 0.0
        return len(rt & et) / len(et)

    @staticmethod
    def _fuzzy_phrase_score(response: str, expected: str) -> float:
        rt = FeatureExtractor._tokens(response)
        et = FeatureExtractor._tokens(expected)
        if not rt or not et:
            return 0.0

        n = len(et)
        window_sizes = {max(1, n - 1), n, n + 1}
        expected_norm = " ".join(et)
        best = 0.0
        for size in window_sizes:
            if size > len(rt):
                continue
            for i in range(0, len(rt) - size + 1):
                candidate = " ".join(rt[i : i + size])
                best = max(best, SequenceMatcher(None, candidate, expected_norm).ratio())
        return best

    @staticmethod
    def value_match_score(response: str, expected: str) -> float:
        """Score whether a response discloses the expected value.

        This is intentionally lightweight: exact phrase match first, then token
        overlap and fuzzy phrase matching for minor lexical variation.
        """
        if not response or not expected:
            return 0.0
        normalized_response = FeatureExtractor._normalize(response)
        if FeatureExtractor._has_phrase(normalized_response, expected):
            return 1.0
        overlap = FeatureExtractor._token_overlap_score(normalized_response, expected)
        fuzzy = FeatureExtractor._fuzzy_phrase_score(normalized_response, expected)
        if overlap >= 0.67 and fuzzy >= 0.72:
            return max(overlap, fuzzy)
        return max(overlap * 0.8, fuzzy * 0.65)

    @staticmethod
    def _value_negated(response: str, expected: str) -> bool:
        pattern = FeatureExtractor._phrase_pattern(expected)
        for match in pattern.finditer(response):
            prefix = response[max(0, match.start() - 60) : match.start()]
            suffix = response[match.end() : min(len(response), match.end() + 40)]
            nearby_tokens = FeatureExtractor._tokens(prefix)[-6:] + FeatureExtractor._tokens(suffix)[:3]
            if any(tok in NEGATION_TOKENS for tok in nearby_tokens):
                return True

        rt = FeatureExtractor._tokens(response)
        et = FeatureExtractor._tokens(expected)
        if not rt or not et:
            return False

        n = len(et)
        for i in range(0, len(rt) - n + 1):
            if rt[i : i + n] != et:
                continue
            left = rt[max(0, i - 5) : i]
            right = rt[i + n : min(len(rt), i + n + 3)]
            if any(tok in NEGATION_TOKENS for tok in left + right):
                return True
        return False

    @staticmethod
    def _source_score(response: str) -> float:
        if FeatureExtractor._has_any_phrase(response, UNKNOWN_PHRASES):
            return 0.0
        if FeatureExtractor._has_any_phrase(response, SOURCE_MEMORY_PHRASES):
            return 1.0
        if FeatureExtractor._has_any_phrase(response, SOURCE_INFERRED_PHRASES):
            return 0.4
        return 0.0

    @staticmethod
    def parse_response(response: str, expected_value: str) -> ResponseSignals:
        normalized = FeatureExtractor._normalize(response)
        value_score = FeatureExtractor.value_match_score(normalized, expected_value)
        value_present = value_score >= 0.55
        value_negated = FeatureExtractor._value_negated(normalized, expected_value)
        unknown = FeatureExtractor._has_any_phrase(normalized, UNKNOWN_PHRASES)
        denied = FeatureExtractor._has_any_pattern(normalized, DENIAL_PATTERNS)
        affirmed = FeatureExtractor._has_any_pattern(normalized, AFFIRMATION_PATTERNS)
        hedged = FeatureExtractor._has_any_phrase(normalized, HEDGING_PHRASES)
        source_score = FeatureExtractor._source_score(normalized)
        return ResponseSignals(
            value_score=value_score,
            value_present=value_present,
            value_negated=value_negated,
            unknown=unknown,
            affirmed=affirmed,
            denied=denied,
            hedged=hedged,
            source_score=source_score,
        )

    @staticmethod
    def _score_recall(signals: ResponseSignals) -> float:
        if signals.unknown or signals.value_negated:
            return 0.0
        if signals.value_score >= 0.9:
            return 0.75 if signals.hedged else 1.0
        if signals.value_score >= 0.5:
            return 0.4 if signals.hedged else 0.6
        return 0.0

    @staticmethod
    def _score_indirect(signals: ResponseSignals) -> float:
        return FeatureExtractor._score_recall(signals)

    @staticmethod
    def _score_provenance(signals: ResponseSignals) -> float:
        disclosure = FeatureExtractor._score_recall(signals)
        if disclosure == 0.0:
            return 0.0
        return min(1.0, 0.7 * disclosure + 0.3 * signals.source_score)

    @staticmethod
    def _score_confirmation(signals: ResponseSignals) -> float:
        if signals.unknown or signals.value_negated or signals.denied:
            return 0.0
        if signals.affirmed and signals.value_score >= 0.5:
            return 1.0
        if signals.affirmed:
            return 0.7
        return 0.0

    @staticmethod
    def score_response(response: str, expected_value: str, probe_type: ProbeType) -> float:
        signals = FeatureExtractor.parse_response(response, expected_value)
        if probe_type == ProbeType.CONFIRMATION:
            return FeatureExtractor._score_confirmation(signals)
        if probe_type == ProbeType.PROVENANCE:
            return FeatureExtractor._score_provenance(signals)
        if probe_type == ProbeType.INDIRECT_REASONING:
            return FeatureExtractor._score_indirect(signals)
        return FeatureExtractor._score_recall(signals)

    @staticmethod
    def text_similarity(response: str, expected: str) -> float:
        """Compatibility wrapper for older baseline code."""
        return FeatureExtractor.value_match_score(response, expected)

    @staticmethod
    def recalled_memory_contents(result: ProbeResult) -> list[str]:
        metadata = result.memory_metadata or {}
        memories = metadata.get("recalled_memories") or []
        contents = []
        for memory in memories:
            if isinstance(memory, dict):
                content = memory.get("content", "")
            else:
                content = str(memory)
            if content:
                contents.append(content)
        return contents

    @classmethod
    def score_recalled_memories(
        cls,
        result: ProbeResult,
        expected_value: str,
    ) -> float:
        """Score the best recalled memory against an expected key value."""
        scores = [
            cls.score_response(content, expected_value, ProbeType.DIRECT_RECALL)
            for content in cls.recalled_memory_contents(result)
        ]
        return max(scores) if scores else 0.0

    @staticmethod
    def add_memory_statement_features(
        features: dict,
        fact_scores: list[float],
        decoy_scores: list[float],
        scorer: str,
    ) -> dict:
        fact_max = max(fact_scores) if fact_scores else 0.0
        decoy_max = max(decoy_scores) if decoy_scores else 0.0
        fact_mean = sum(fact_scores) / len(fact_scores) if fact_scores else 0.0
        decoy_mean = sum(decoy_scores) / len(decoy_scores) if decoy_scores else 0.0

        features["memory_statement_scorer"] = scorer
        features["fact_memory_statement_scores"] = fact_scores
        features["decoy_memory_statement_scores"] = decoy_scores
        features["fact_memory_statement_score_max"] = fact_max
        features["decoy_memory_statement_score_max"] = decoy_max
        features["fact_memory_statement_score_mean"] = fact_mean
        features["decoy_memory_statement_score_mean"] = decoy_mean
        features["delta_memory_statement_score"] = fact_max - decoy_max
        return features

    def extract_round_features(
        self,
        fact: Fact,
        fact_results: list[ProbeResult],
        decoy_results: list[ProbeResult],
        probe_type: ProbeType,
    ) -> dict:
        fs = [
            self.score_response(r.response, fact.key_value, probe_type)
            for r in fact_results
        ]
        ds = [
            self.score_response(r.response, r.probe.expected_if_member, probe_type)
            for r in decoy_results
        ]
        return self.extract_round_features_from_scores(
            fs,
            ds,
            fact_results,
            decoy_results,
        )

    def extract_round_features_from_scores(
        self,
        fact_scores: list[float],
        decoy_scores: list[float],
        fact_results: list[ProbeResult],
        decoy_results: list[ProbeResult],
    ) -> dict:
        f: dict = {}
        fs = fact_scores
        ds = decoy_scores
        fact_score = sum(fs) / len(fs) if fs else 0.0
        decoy_score = sum(ds) / len(ds) if ds else 0.0
        delta_score = fact_score - decoy_score

        f["fact_response_scores"] = fs
        f["decoy_response_scores"] = ds
        f["fact_response_score_mean"] = fact_score
        f["decoy_response_score_mean"] = decoy_score
        f["delta_response_score"] = delta_score

        # Backward-compatible names used by existing logs and evidence records.
        f["fact_similarity_mean"] = fact_score
        f["decoy_similarity_mean"] = decoy_score
        f["delta_similarity"] = delta_score

        has_memory_metadata = any(
            r.memory_metadata is not None for r in fact_results + decoy_results
        )
        fr = [r for r in fact_results if r.recall_triggered is not None]
        dr = [r for r in decoy_results if r.recall_triggered is not None]
        if fr and dr and not has_memory_metadata:
            f["delta_recall_rate"] = (
                sum(1 for r in fr if r.recall_triggered) / len(fr)
            ) - (
                sum(1 for r in dr if r.recall_triggered) / len(dr)
            )
            f["delta_recall_similarity"] = max(
                (r.recall_top_similarity or 0) for r in fr
            ) - max((r.recall_top_similarity or 0) for r in dr)

        flp = [r.mean_logprob for r in fact_results if r.mean_logprob is not None]
        dlp = [r.mean_logprob for r in decoy_results if r.mean_logprob is not None]
        if flp and dlp:
            raw = sum(flp) / len(flp) - sum(dlp) / len(dlp)
            f["delta_logprob"] = max(-1.0, min(1.0, raw / 5.0))

        return f

    def compute_round_score(self, features: dict) -> float:
        weights = {
            "delta_response_score": 1.0,
            "delta_memory_statement_score": 1.0,
            "delta_logprob": 1.0,
        }
        s, tw = 0.0, 0.0
        for k, w in weights.items():
            if k in features:
                s += w * features[k]
                tw += w
        return s / tw if tw > 0 else 0.0

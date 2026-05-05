"""Associative Recall for PMC — v4 (all patches applied).

Changes from v1:
1. Search ALL episodes (not just recent 20) — old facts remain reachable
2. Relevance-first scoring via tuple (rel, strength) — no more strength-dominated ranking
3. Larger topK: episodic 10, semantic 10, procedural 5
4. Tokenize supports numbers: "3090", "0.3", "72b" are now tokens
5. Lower min_strength: 0.02 (aligned with store defaults)
6. Minimal stopwords — removed "user", "agent", "like" etc. that can be meaningful
7. rel > 0 threshold (aligned with basebot) — any keyword match is a candidate
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from nanobot.memory.models import EpisodicMemory, ProceduralMemory, SemanticMemory
from nanobot.memory.store import MemoryStore


# Minimal stopwords — only truly content-free function words
STOPWORDS = {
    "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can",
    "what", "which", "who", "whom", "whose", "where", "when", "how", "why",
    "that", "this", "these", "those", "it", "its",
    "an", "and", "or", "but", "not", "no", "nor",
    "for", "with", "about", "from", "into", "through", "during",
    "before", "after", "above", "below", "between", "under", "over",
    "if", "then", "than", "so", "as", "at", "by", "in", "on", "of", "to", "up",
    "he", "she", "they", "we", "you", "me", "him", "her", "us", "them",
    "my", "your", "his", "our", "their",
    "am", "just", "also", "very", "much", "more", "most", "some", "any",
    "all", "each", "every", "both", "few", "many", "such",
    "here", "there", "now", "still", "already", "again",
}


@dataclass
class RecallResult:
    """Memories retrieved for the current context."""

    episodic: list[EpisodicMemory]
    semantic: list[SemanticMemory]
    procedural: list[ProceduralMemory]

    def is_empty(self) -> bool:
        return not self.episodic and not self.semantic and not self.procedural

    def format_for_prompt(self) -> str:
        if self.is_empty():
            return ""

        parts = []

        if self.procedural:
            items = [
                f"  - {p.trigger} -> {p.action} (confidence: {p.confidence:.0%})"
                for p in self.procedural
            ]
            parts.append("**Strategies (procedural memory):**\n" + "\n".join(items))

        if self.semantic:
            items = [
                f"  - {s.content} (confidence: {s.confidence:.0%}, verified {s.reinforcement_count}x)"
                for s in self.semantic
            ]
            parts.append("**Knowledge (semantic memory):**\n" + "\n".join(items))

        if self.episodic:
            items = [
                f"  - [{e.outcome}] {e.query[:80]} -> {e.summary[:120]}"
                for e in self.episodic
            ]
            parts.append("**Recent experience (episodic memory):**\n" + "\n".join(items))

        return "\n\n".join(parts)


def _tokenize(text: str) -> set[str]:
    """Tokenize with stopword removal. Supports alphanumeric tokens (3090, 0.3, v2, etc.)."""
    # Match: words, numbers, alphanumeric combos, Chinese chars
    raw = re.findall(r"[a-z0-9][a-z0-9.]*[a-z0-9]|[a-z]{2,}|[\u4e00-\u9fff]{2,}", text.lower())
    return {t for t in raw if len(t) >= 2 and t not in STOPWORDS}


def _relevance(query_tokens: set[str], text: str, tags: list[str]) -> float:
    """Compute keyword overlap relevance (0-1)."""
    text_tokens = _tokenize(text)
    tag_tokens = {t.lower() for t in tags} - STOPWORDS
    all_tokens = text_tokens | tag_tokens
    if not query_tokens or not all_tokens:
        return 0.0
    overlap = len(query_tokens & all_tokens)
    return overlap / max(len(query_tokens), 1)


class Recall:
    """Associative recall across all memory tiers."""

    def __init__(self, store: MemoryStore):
        self.store = store

    def recall(
        self,
        query: str,
        max_episodic: int = 10,
        max_semantic: int = 10,
        max_procedural: int = 5,
        min_strength: float = 0.02,
        update_access: bool = True,
    ) -> RecallResult:
        """Retrieve the most relevant memories for a query.

        Key design decisions:
        - Search ALL episodes (not just recent N) — prevents historical fact loss
        - Scoring: (relevance, strength) tuple — relevance dominates, strength breaks ties
        - Any match with rel > 0 is a candidate — aligned with basebot's approach
        - Larger topK to prevent correct evidence falling outside cutoff
        """
        query_tokens = _tokenize(query)

        # ── Episodic: search ALL episodes ──
        all_episodes = self.store.get_recent_episodes(9999)
        ep_scored = []
        for e in all_episodes:
            rel = _relevance(query_tokens, f"{e.query} {e.summary}", e.tags)
            if rel > 0:
                ep_scored.append(((rel, e.strength()), e))
        ep_scored.sort(key=lambda x: x[0], reverse=True)
        if update_access:
            for _, e in ep_scored[:max_episodic]:
                e.touch()
                self.store.save_episode(e)

        # ── Semantic: search all active ──
        sem_scored = []
        for s in self.store.get_active_semantic(min_strength):
            rel = _relevance(query_tokens, s.content, s.tags)
            if rel > 0:
                sem_scored.append(((rel, s.strength()), s))
        sem_scored.sort(key=lambda x: x[0], reverse=True)
        if update_access:
            for _, s in sem_scored[:max_semantic]:
                s.touch()
                self.store.save_semantic(s)

        # ── Procedural: search all active ──
        proc_scored = []
        for p in self.store.get_active_procedural(min_strength):
            rel = _relevance(query_tokens, f"{p.trigger} {p.action}", p.tags)
            if rel > 0:
                proc_scored.append(((rel, p.strength()), p))
        proc_scored.sort(key=lambda x: x[0], reverse=True)
        if update_access:
            for _, p in proc_scored[:max_procedural]:
                p.touch()
                self.store.save_procedural(p)

        return RecallResult(
            episodic=[e for _, e in ep_scored[:max_episodic]],
            semantic=[s for _, s in sem_scored[:max_semantic]],
            procedural=[p for _, p in proc_scored[:max_procedural]],
        )

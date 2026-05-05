"""Memory Extractor for CEA-MIA.

Extracts dialogue memory units and splits them by conversation/session group.
For perltqa, dialogue facts are extracted from raw dialogue events. For LOCOMO,
the dataset-provided session observations are used directly as memory units.
All facts from the same event/session group are assigned to the same membership
side.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
from pathlib import Path
from typing import Optional

import httpx

from config import Config
from memory_unit import MemoryUnit, MemoryDataset, UserMemorySet, PerltType


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("memory_extractor")


# ── LLM Extraction Prompts ────────────────────────────────────────────────

EXTRACT_PROFILE_PROMPT = """\
Extract factual statements from this profile data. Each fact should be a standalone sentence.

Profile:
{text}

Return JSON array of facts:
[{{"content": "fact statement"}}]

Example:
[{{"content": "Wang Xiaoming is 28 years old"}}]
"""

EXTRACT_EVENT_PROMPT = """\
Extract key factual statements from this event description. Focus on:
- Who was involved
- What happened
- When and where

Event:
{text}

Return JSON array of facts:
[{{"content": "fact statement"}}]
"""

EXTRACT_DIALOGUE_PROMPT = """\
Extract factual information revealed in this dialogue. Focus on personal facts,
preferences, plans, and relationships mentioned.

Dialogue:
{text}

Return JSON array of facts:
[{{"content": "fact statement"}}]
"""

EXTRACT_RELATIONSHIP_PROMPT = """\
Extract factual statements about this relationship.

Relationship:
{text}

Return JSON array of facts:
[{{"content": "fact statement"}}]
"""

EXTRACT_MSC_SESSION_PAIR_PROMPT = """\
Extract memory statements from two MSC conversations between the same two speakers.

Important labels:
- Speaker 1 and Speaker 2 are the two conversation partners.
- Write every memory as a standalone third-person sentence, starting with either
  "Speaker 1" or "Speaker 2".
- Each memory must be understandable without reading the dialogue and should contain
  one clear fact only.
- Extract personal facts, preferences, plans, relationships, jobs, hobbies, locations,
  experiences, and stable contextual facts that a memory system could store.
- Do not include generic chit-chat, questions, jokes without factual content, or facts
  about the outside world.
- Session 2 memories must be incremental: do NOT repeat facts that are already stated
  in Session 1, even if the wording is different.

Session 1 dialogue:
{session1}

Session 2 dialogue:
{session2}

Return ONLY valid JSON in this exact shape:
{{
  "session1_memories": [
    {{"content": "Speaker 1 ..."}},
    {{"content": "Speaker 2 ..."}}
  ],
  "session2_memories": [
    {{"content": "Speaker 1 ..."}},
    {{"content": "Speaker 2 ..."}}
  ]
}}
"""


class MemoryExtractor:
    """Extract and split memory units from supported memory datasets."""

    def __init__(
        self,
        dataset_name: str = "perltqa",
        perlt_types: Optional[list[PerltType]] = None,
        seed: int = 42,
        split_ratio: float = 0.5,
        max_users: Optional[int] = None,
        max_concurrency: int = 10,
        config: Optional[Config] = None,
    ):
        """
        Args:
            dataset_name: Name of dataset, maps to rawdata/{dataset_name}/
            perlt_types: Deprecated; extraction is dialogue-only.
            seed: Random seed for reproducible splits
            split_ratio: Fraction to use as members (default: 0.5)
            max_users: Limit number of users to process (for testing)
            max_concurrency: Max parallel LLM calls for dialogue extraction
            config: Config object for LLM settings
        """
        self.dataset_name = dataset_name
        if self.dataset_name == "msc" and max_users is None:
            max_users = 50
        if perlt_types and perlt_types != [PerltType.DIALOGUE]:
            log.warning(
                "Ignoring perlt_types=%s; memory extraction is dialogue-only.",
                [t.value for t in perlt_types],
            )
        self.perlt_types = [PerltType.DIALOGUE]
        self.seed = seed
        self.split_ratio = split_ratio
        self.max_users = max_users
        self.max_concurrency = max_concurrency
        self.config = config or Config()

        self.rng = random.Random(seed)
        self.data_dir = Path(__file__).parent / "rawdata" / dataset_name

        # HTTP client for LLM calls
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    # ── Main extraction entry point ───────────────────────────────────────

    async def extract(self) -> MemoryDataset:
        """Extract and split memory units from the dataset.

        Split strategy: For each user, group extracted dialogue facts by
        dialogue event id and split those groups. A single event id is never
        split across member and non-member memories.
        """
        log.info("Loading dataset: %s", self.dataset_name)
        log.info("Types: %s", [t.value for t in self.perlt_types])
        log.info("Seed: %d, Split ratio: %.2f", self.seed, self.split_ratio)

        # Check if LLM is required but not configured. LOCOMO uses provided
        # observations directly and does not require LLM extraction.
        if self.dataset_name != "locomo" and not self.config.api_base:
            log.warning("=" * 60)
            log.warning("LLM required for dialogue extraction but not configured!")
            log.warning("Set CEA_MI_API_BASE to your vLLM endpoint (e.g., http://localhost:8000/v1)")
            log.warning("Dialogue extraction will be skipped.")
            log.warning("=" * 60)

        # Load raw data
        raw_data = self._load_raw_data()
        if self.max_users is not None:
            self.rng.shuffle(raw_data)
            raw_data = raw_data[:self.max_users]
        log.info("Loaded %d users", len(raw_data))

        if self.dataset_name == "locomo":
            return self._extract_locomo_observations(raw_data)
        if self.dataset_name == "msc":
            return await self._extract_msc_session_pair_memories(raw_data)

        # Collect dialogue items only. Each raw dialogue carries an event id,
        # and split labels are assigned at that event-id level after extraction.
        all_items: list[tuple[int, str, PerltType, dict | str]] = []

        for user_idx, user_data in enumerate(raw_data):
            dialogues = user_data.get("dialogues", {})
            for dial_key, dial_data in dialogues.items():
                all_items.append((user_idx, dial_key, PerltType.DIALOGUE, dial_data))

        log.info("Found %d extractable items", len(all_items))

        # First extract ALL units (without membership label)
        all_units = await self._extract_from_items(
            all_items, is_member=False, max_concurrency=self.max_concurrency
        )
        log.info("Extracted %d total memory units", len(all_units))

        # Group units by user_id first, then by dialogue event_id within each user.
        user_groups: dict[int, dict[str, list[MemoryUnit]]] = {}
        for unit in all_units:
            event_id = unit.event_id or self._event_id_from_source_key(unit.source_key)
            unit.event_id = event_id
            user_groups.setdefault(unit.user_id, {}).setdefault(event_id, []).append(unit)

        log.info("Grouped into %d users", len(user_groups))

        # Build per-user memory sets with stratified splits
        user_memory_sets: dict[int, UserMemorySet] = {}

        for user_id in sorted(user_groups.keys()):
            user_members = []
            user_non_members = []

            event_groups = list(user_groups[user_id].items())
            self.rng.shuffle(event_groups)
            split_idx = int(len(event_groups) * self.split_ratio)
            if len(event_groups) > 1:
                split_idx = max(1, min(split_idx, len(event_groups) - 1))

            for group_idx, (event_id, units) in enumerate(event_groups):
                is_member_group = group_idx < split_idx

                # Handle edge case: if only 1 event group, randomly assign it.
                if len(event_groups) == 1:
                    is_member_group = self.rng.random() < self.split_ratio

                for unit in units:
                    unit.is_member = is_member_group

                if is_member_group:
                    user_members.extend(units)
                else:
                    user_non_members.extend(units)

                log.debug(
                    "  User %d, event_id=%s: %d units -> %s",
                    user_id,
                    event_id,
                    len(units),
                    "member" if is_member_group else "non-member",
                )

            if not user_members or not user_non_members:
                log.warning(
                    "  User %d has one-sided split after event grouping: %d members, %d non-members",
                    user_id,
                    len(user_members),
                    len(user_non_members),
                )

            user_memory_sets[user_id] = UserMemorySet(
                user_id=user_id,
                members=user_members,
                non_members=user_non_members,
            )
            log.info("  User %d: %d members, %d non-members",
                     user_id, len(user_members), len(user_non_members))

        dataset = MemoryDataset(
            users=user_memory_sets,
            dataset_name=self.dataset_name,
            seed=self.seed,
            perlt_types=self.perlt_types,
        )

        log.info("Total: %d members, %d non-members across %d users",
                 len(dataset.all_members), len(dataset.all_non_members), len(dataset.users))

        log.info("Dataset stats: %s", dataset.stats())
        return dataset

    # ── Data loading ──────────────────────────────────────────────────────

    def _load_raw_data(self) -> list[dict]:
        """Load the raw dataset file."""
        if self.dataset_name == "locomo":
            for name in ("locomo10.json", "locomo.json"):
                data_file = self.data_dir / name
                if data_file.exists():
                    with open(data_file, encoding="utf-8") as f:
                        return json.load(f)
            raise FileNotFoundError(
                f"Dataset file not found: {self.data_dir / 'locomo10.json'}"
            )

        if self.dataset_name == "msc":
            return self._load_msc_session2_records()

        mem_file = self.data_dir / "perltmem_en.json"
        if not mem_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {mem_file}")

        with open(mem_file, encoding="utf-8") as f:
            return json.load(f)

    def _load_msc_session2_records(self) -> list[dict]:
        """Load MSC session_2 JSONL rows.

        Each row contains the current session-2 dialogue plus one previous
        dialogue, which is the original PersonaChat/session-1 conversation.
        """
        session_dir = self.data_dir / "msc" / "msc" / "msc_dialogue" / "session_2"
        if not session_dir.exists():
            raise FileNotFoundError(f"MSC session_2 directory not found: {session_dir}")

        records: list[dict] = []
        for split in ("train", "valid", "test"):
            path = session_dir / f"{split}.txt"
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    record["_msc_split"] = split
                    record["_msc_line_no"] = line_no
                    records.append(record)
        return records

    # ── LOCOMO observation extraction ─────────────────────────────────────

    @staticmethod
    def _sample_id_to_user_id(sample_id: str, fallback_idx: int) -> int:
        """Map LOCOMO sample ids like ``conv-26`` to stable integer user ids."""
        if sample_id:
            suffix = sample_id.rsplit("-", 1)[-1]
            if suffix.isdigit():
                return int(suffix)
        return fallback_idx

    @staticmethod
    def _locomo_session_sort_key(session_key: str) -> tuple[int, str]:
        """Sort ``session_10`` after ``session_9``."""
        stem = session_key.replace("_observation", "")
        suffix = stem.rsplit("_", 1)[-1]
        return (int(suffix), stem) if suffix.isdigit() else (10**9, stem)

    def _extract_locomo_observations(self, raw_data: list[dict]) -> MemoryDataset:
        """Use LOCOMO session observations directly as MemoryUnits.

        Each ``sample_id`` is treated as one user. Within a sample, all
        observations from the same session are assigned together to either the
        member or non-member side.
        """
        user_memory_sets: dict[int, UserMemorySet] = {}

        for sample_idx, sample in enumerate(raw_data):
            sample_id = str(sample.get("sample_id") or f"sample_{sample_idx}")
            user_id = self._sample_id_to_user_id(sample_id, sample_idx)
            observations = sample.get("observation") or {}

            session_groups: list[tuple[str, list[MemoryUnit]]] = []
            for obs_key in sorted(observations.keys(), key=self._locomo_session_sort_key):
                session_obs = observations.get(obs_key) or {}
                if not isinstance(session_obs, dict):
                    continue

                session_id = obs_key.replace("_observation", "")
                event_id = f"{sample_id}:{session_id}"
                units: list[MemoryUnit] = []

                for speaker, speaker_observations in session_obs.items():
                    if not isinstance(speaker_observations, list):
                        continue
                    for obs_idx, observation in enumerate(speaker_observations):
                        if not isinstance(observation, list) or not observation:
                            continue
                        content = str(observation[0]).strip()
                        if not content:
                            continue
                        evidence = str(observation[1]).strip() if len(observation) > 1 else str(obs_idx)
                        units.append(MemoryUnit(
                            content=content,
                            perlt_type=PerltType.DIALOGUE,
                            user_id=user_id,
                            source_key=f"{sample_id}:{session_id}:{speaker}:{evidence}",
                            event_id=event_id,
                        ))

                if units:
                    session_groups.append((event_id, units))

            self.rng.shuffle(session_groups)
            split_idx = int(len(session_groups) * self.split_ratio)
            if len(session_groups) > 1:
                split_idx = max(1, min(split_idx, len(session_groups) - 1))

            user_members: list[MemoryUnit] = []
            user_non_members: list[MemoryUnit] = []

            for group_idx, (event_id, units) in enumerate(session_groups):
                is_member_group = group_idx < split_idx
                if len(session_groups) == 1:
                    is_member_group = self.rng.random() < self.split_ratio

                for unit in units:
                    unit.is_member = is_member_group

                if is_member_group:
                    user_members.extend(units)
                else:
                    user_non_members.extend(units)

                log.debug(
                    "  LOCOMO %s, event_id=%s: %d units -> %s",
                    sample_id,
                    event_id,
                    len(units),
                    "member" if is_member_group else "non-member",
                )

            if not user_members or not user_non_members:
                log.warning(
                    "  LOCOMO sample %s has one-sided split: %d members, %d non-members",
                    sample_id,
                    len(user_members),
                    len(user_non_members),
                )

            user_memory_sets[user_id] = UserMemorySet(
                user_id=user_id,
                members=user_members,
                non_members=user_non_members,
            )
            log.info(
                "  LOCOMO sample %s -> user_id=%d: %d members, %d non-members",
                sample_id,
                user_id,
                len(user_members),
                len(user_non_members),
            )

        dataset = MemoryDataset(
            users=user_memory_sets,
            dataset_name=self.dataset_name,
            seed=self.seed,
            perlt_types=self.perlt_types,
        )

        log.info(
            "Total: %d members, %d non-members across %d users",
            len(dataset.all_members),
            len(dataset.all_non_members),
            len(dataset.users),
        )
        log.info("Dataset stats: %s", dataset.stats())
        return dataset

    # ── MSC session-pair extraction ───────────────────────────────────────

    @staticmethod
    def _msc_initial_data_id(record: dict, fallback: str = "") -> str:
        metadata = record.get("metadata") or {}
        return (
            str(metadata.get("initial_data_id") or record.get("initial_data_id") or fallback)
            .strip()
        )

    @staticmethod
    def _msc_turn_speaker(turn: dict, turn_idx: int) -> str:
        raw_id = str(turn.get("id") or "").strip().lower()
        if raw_id in {"speaker 1", "bot_0", "0", "speaker_1"}:
            return "Speaker 1"
        if raw_id in {"speaker 2", "bot_1", "1", "speaker_2"}:
            return "Speaker 2"
        # PersonaChat rows in previous_dialogs often omit speaker ids.
        return "Speaker 1" if turn_idx % 2 == 0 else "Speaker 2"

    @classmethod
    def _format_msc_dialogue(cls, dialog: list[dict], max_turns: int = 24) -> str:
        lines: list[str] = []
        for idx, turn in enumerate(dialog[:max_turns]):
            text = str(turn.get("text") or "").strip()
            if not text:
                continue
            speaker = cls._msc_turn_speaker(turn, idx)
            lines.append(f"{speaker}: {text}")
        return "\n".join(lines)

    @staticmethod
    def _normalize_memory_text(text: str) -> str:
        text = " ".join(str(text).lower().split())
        return "".join(ch for ch in text if ch.isalnum() or ch.isspace()).strip()

    @staticmethod
    def _coerce_memory_list(value) -> list[str]:
        if not isinstance(value, list):
            return []
        memories: list[str] = []
        for item in value:
            if isinstance(item, dict):
                content = str(item.get("content") or "").strip()
            else:
                content = str(item or "").strip()
            if content:
                memories.append(content)
        return memories

    async def _extract_msc_session_pair_memories(self, raw_data: list[dict]) -> MemoryDataset:
        """Extract MSC memories from session 1 and session 2 in one LLM call.

        Each MSC row is treated as one user in this benchmark: a fixed pair of
        conversation partners. Session-1 memories are members; incremental
        session-2 memories are non-members.
        """
        if not self.config.api_base:
            log.warning("Skipping MSC extraction: LLM API not configured")
            return MemoryDataset(dataset_name=self.dataset_name, seed=self.seed)

        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def extract_one(user_idx: int, record: dict) -> Optional[UserMemorySet]:
            async with semaphore:
                return await self._extract_single_msc_pair(user_idx, record)

        tasks = [extract_one(user_idx, record) for user_idx, record in enumerate(raw_data)]
        results = await asyncio.gather(*tasks)

        user_memory_sets: dict[int, UserMemorySet] = {}
        for user_set in results:
            if user_set is not None:
                user_memory_sets[user_set.user_id] = user_set

        dataset = MemoryDataset(
            users=user_memory_sets,
            dataset_name=self.dataset_name,
            seed=self.seed,
            perlt_types=self.perlt_types,
        )

        log.info(
            "MSC total: %d members, %d non-members across %d pair-users",
            len(dataset.all_members),
            len(dataset.all_non_members),
            len(dataset.users),
        )
        log.info("Dataset stats: %s", dataset.stats())
        return dataset

    async def _extract_single_msc_pair(
        self,
        user_idx: int,
        record: dict,
    ) -> Optional[UserMemorySet]:
        conversation_id = self._msc_initial_data_id(record, fallback=f"msc_{user_idx}")
        previous_dialogs = record.get("previous_dialogs") or []
        if not previous_dialogs:
            log.warning("Skipping MSC %s: missing previous_dialogs", conversation_id)
            return None

        session1_dialog = previous_dialogs[0].get("dialog") or []
        session2_dialog = record.get("dialog") or []
        session1_text = self._format_msc_dialogue(session1_dialog)
        session2_text = self._format_msc_dialogue(session2_dialog)
        if not session1_text or not session2_text:
            log.warning("Skipping MSC %s: empty session text", conversation_id)
            return None

        prompt = EXTRACT_MSC_SESSION_PAIR_PROMPT.format(
            session1=session1_text[:3500],
            session2=session2_text[:3500],
        )
        payload = await self._call_llm_json(prompt, max_tokens=1536)

        session1_memories = self._coerce_memory_list(payload.get("session1_memories"))
        session2_memories = self._coerce_memory_list(payload.get("session2_memories"))

        members: list[MemoryUnit] = []
        non_members: list[MemoryUnit] = []
        seen_session1: set[str] = set()
        seen_session2: set[str] = set()

        for idx, content in enumerate(session1_memories):
            key = self._normalize_memory_text(content)
            if not key or key in seen_session1:
                continue
            seen_session1.add(key)
            members.append(MemoryUnit(
                content=content,
                perlt_type=PerltType.DIALOGUE,
                user_id=user_idx,
                source_key=f"msc:{conversation_id}:session_1#{idx}",
                event_id=f"msc:{conversation_id}:session_1",
                is_member=True,
            ))

        for idx, content in enumerate(session2_memories):
            key = self._normalize_memory_text(content)
            if not key or key in seen_session2 or key in seen_session1:
                continue
            seen_session2.add(key)
            non_members.append(MemoryUnit(
                content=content,
                perlt_type=PerltType.DIALOGUE,
                user_id=user_idx,
                source_key=f"msc:{conversation_id}:session_2#{idx}",
                event_id=f"msc:{conversation_id}:session_2",
                is_member=False,
            ))

        if not members or not non_members:
            log.warning(
                "Skipping MSC %s: extracted one-sided memories (%d members, %d non-members)",
                conversation_id,
                len(members),
                len(non_members),
            )
            return None

        log.info(
            "  MSC %s -> user_id=%d: %d session1 members, %d session2 non-members",
            conversation_id,
            user_idx,
            len(members),
            len(non_members),
        )
        return UserMemorySet(user_id=user_idx, members=members, non_members=non_members)

    # ── Extraction logic ──────────────────────────────────────────────────

    async def _extract_from_items(
        self,
        items: list[tuple[int, str, PerltType, dict | str]],
        is_member: bool,
        max_concurrency: int = 10,
    ) -> list[MemoryUnit]:
        """Extract MemoryUnits from a list of items with parallel processing.

        Args:
            items: List of (user_idx, source_key, perlt_type, data) tuples
            is_member: Whether these are member items
            max_concurrency: Max parallel extractions (for LLM rate limiting)
        """
        # Separate sync and async items
        sync_items = []  # profile, social_relationship - no LLM needed
        async_items = []  # event, dialogue - may need LLM

        for item in items:
            user_idx, source_key, perlt_type, data = item
            if perlt_type in (PerltType.PROFILE, PerltType.SOCIAL_RELATIONSHIP):
                sync_items.append(item)
            else:
                async_items.append(item)

        units = []

        # Process sync items directly (fast, no I/O)
        for user_idx, source_key, perlt_type, data in sync_items:
            try:
                if perlt_type == PerltType.PROFILE:
                    extracted = self._extract_profile_direct(user_idx, source_key, data)
                else:
                    extracted = self._extract_relationship_direct(user_idx, source_key, data)
                for unit in extracted:
                    unit.is_member = is_member
                units.extend(extracted)
            except Exception as e:
                log.warning("Failed to extract %s/%s: %s", user_idx, source_key, e)

        # Process async items with concurrency control
        if async_items:
            semaphore = asyncio.Semaphore(max_concurrency)

            async def extract_with_limit(item):
                user_idx, source_key, perlt_type, data = item
                async with semaphore:
                    try:
                        return await self._extract_single(user_idx, source_key, perlt_type, data)
                    except Exception as e:
                        log.warning("Failed to extract %s/%s: %s", user_idx, source_key, e)
                        return []

            # Run all async extractions concurrently
            results = await asyncio.gather(*[extract_with_limit(item) for item in async_items])

            for extracted in results:
                for unit in extracted:
                    unit.is_member = is_member
                units.extend(extracted)

        return units

    async def _extract_single(
        self,
        user_idx: int,
        source_key: str,
        perlt_type: PerltType,
        data: dict | str,
    ) -> list[MemoryUnit]:
        """Extract MemoryUnits from a single item."""

        if perlt_type == PerltType.PROFILE:
            return self._extract_profile_direct(user_idx, source_key, data)
        elif perlt_type == PerltType.SOCIAL_RELATIONSHIP:
            return self._extract_relationship_direct(user_idx, source_key, data)
        elif perlt_type == PerltType.EVENT:
            return await self._extract_event_llm(user_idx, source_key, data)
        elif perlt_type == PerltType.DIALOGUE:
            return await self._extract_dialogue_llm(user_idx, source_key, data)
        else:
            return []

    @staticmethod
    def _event_id_from_source_key(source_key: str) -> str:
        """Recover a dialogue event id from a source key like ``1_0_0#3``."""
        source_key = source_key or ""
        return source_key.split("#", 1)[0] or source_key or "unknown"

    @staticmethod
    def _normalize_event_id(event_ref, fallback_source_key: str) -> str:
        """Normalize the raw dialogue ``events`` field into a split group key."""
        if isinstance(event_ref, list):
            parts = [str(part).strip() for part in event_ref if str(part).strip()]
            if parts:
                return "+".join(parts)
        elif event_ref:
            return str(event_ref).strip()
        return MemoryExtractor._event_id_from_source_key(fallback_source_key)

    # ── Direct extraction (no LLM needed) ─────────────────────────────────

    def _extract_profile_direct(
        self, user_idx: int, source_key: str, profile: dict
    ) -> list[MemoryUnit]:
        """Extract profile facts directly (structured data)."""
        units = []

        # Get protagonist name for context
        name = profile.get("Protagonist", "The user")

        for field, value in profile.items():
            if field == "Protagonist":
                continue  # Skip name itself
            if not value or not str(value).strip():
                continue

            content = f"{name}'s {field} is {value}"
            units.append(MemoryUnit(
                content=content,
                perlt_type=PerltType.PROFILE,
                user_id=user_idx,
                source_key=source_key,
            ))

        return units

    def _extract_relationship_direct(
        self, user_idx: int, source_key: str, rel_data: dict
    ) -> list[MemoryUnit]:
        """Extract relationship facts directly."""
        units = []

        name = rel_data.get("Supporting Characters", "")
        relationship = rel_data.get("Relationship", "")
        description = rel_data.get("Description", "")

        if name and relationship:
            content = f"{name} is the user's {relationship}. {description}"
            units.append(MemoryUnit(
                content=content,
                perlt_type=PerltType.SOCIAL_RELATIONSHIP,
                user_id=user_idx,
                source_key=source_key,
            ))

        return units

    # ── LLM-based extraction ──────────────────────────────────────────────

    async def _extract_event_llm(
        self, user_idx: int, source_key: str, event_data: dict
    ) -> list[MemoryUnit]:
        """Extract event facts using LLM (required)."""
        content = event_data.get("content", "")
        summary = event_data.get("summary", "")
        theme = event_data.get("Theme", "")

        if not content:
            return []

        # LLM is required for event extraction
        if not self.config.api_base:
            log.warning("Skipping event %s: LLM not configured (set CEA_MI_API_BASE)", source_key)
            return []

        units = []
        try:
            extracted = await self._call_llm_extract(content, PerltType.EVENT)
            for fact in extracted:
                fact_content = fact.get("content", "").strip()
                if not fact_content:
                    continue
                units.append(MemoryUnit(
                    content=fact_content,
                    perlt_type=PerltType.EVENT,
                    user_id=user_idx,
                    source_key=source_key,
                ))
            if units:
                log.debug("Extracted %d facts from event %s", len(units), source_key)
            else:
                log.warning("No facts extracted from event %s", source_key)
        except Exception as e:
            log.warning("LLM extraction failed for event %s: %s", source_key, e)

        return units

    async def _extract_dialogue_llm(
        self, user_idx: int, source_key: str, dial_data: dict
    ) -> list[MemoryUnit]:
        """Extract dialogue facts using LLM (required)."""
        event_ref = dial_data.get("events", "")
        event_id = self._normalize_event_id(event_ref, source_key)
        contents = dial_data.get("contents", {})

        if not contents:
            return []

        # LLM is required for dialogue extraction
        if not self.config.api_base:
            log.warning("Skipping dialogue %s: LLM not configured (set CEA_MI_API_BASE)", source_key)
            return []

        # Flatten dialogue turns
        all_turns = []
        for turns in contents.values():
            all_turns.extend(turns)

        dialogue_text = "\n".join(all_turns[:30])  # Limit to 30 turns

        units = []
        try:
            extracted = await self._call_llm_extract(dialogue_text, PerltType.DIALOGUE)
            for fact in extracted:
                fact_content = fact.get("content", "").strip()
                if not fact_content:
                    continue
                units.append(MemoryUnit(
                    content=fact_content,
                    perlt_type=PerltType.DIALOGUE,
                    user_id=user_idx,
                    source_key=source_key,
                    event_id=event_id,
                ))
            if units:
                log.debug("Extracted %d facts from dialogue %s", len(units), source_key)
            else:
                log.warning("No facts extracted from dialogue %s", source_key)
        except Exception as e:
            log.warning("LLM extraction failed for dialogue %s: %s", source_key, e)

        return units

    async def _call_llm_extract(self, text: str, perlt_type: PerltType) -> list[dict]:
        """Call LLM to extract facts from text."""
        prompt_templates = {
            PerltType.PROFILE: EXTRACT_PROFILE_PROMPT,
            PerltType.SOCIAL_RELATIONSHIP: EXTRACT_RELATIONSHIP_PROMPT,
            PerltType.EVENT: EXTRACT_EVENT_PROMPT,
            PerltType.DIALOGUE: EXTRACT_DIALOGUE_PROMPT,
        }

        prompt = prompt_templates[perlt_type].format(text=text[:2000])

        try:
            resp = await self.client.post(
                f"{self.config.api_base}/chat/completions",
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                json={
                    "model": self.config.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 1024,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]

            # Parse JSON from response
            # Handle potential markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            return json.loads(content.strip())

        except httpx.ConnectError as e:
            log.warning("Cannot connect to LLM server at %s: %s", self.config.api_base, e)
            return []
        except Exception as e:
            log.warning("LLM call failed: %s", e)
            return []

    async def _call_llm_json(self, prompt: str, max_tokens: int = 1024) -> dict:
        """Call LLM and parse a JSON object response."""
        try:
            resp = await self.client.post(
                f"{self.config.api_base}/chat/completions",
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                json={
                    "model": self.config.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": max_tokens,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]

            if "```json" in content:
                content = content.split("```json", 1)[1].split("```", 1)[0]
            elif "```" in content:
                content = content.split("```", 1)[1].split("```", 1)[0]

            parsed = json.loads(content.strip())
            return parsed if isinstance(parsed, dict) else {}

        except httpx.ConnectError as e:
            log.warning("Cannot connect to LLM server at %s: %s", self.config.api_base, e)
            return {}
        except Exception as e:
            log.warning("LLM JSON call failed: %s", e)
            return {}

    # ── Save/Load methods ─────────────────────────────────────────────────

    def save(self, dataset: MemoryDataset, output_path: Optional[str | Path] = None) -> Path:
        """Save extracted dataset to JSON file.

        Args:
            dataset: The MemoryDataset to save
            output_path: Optional path. If not provided, auto-generates based on params.

        Returns:
            Path to the saved file
        """
        if output_path is None:
            # Auto-generate the canonical dataset filename used by attacks.
            filename = f"{self.dataset_name}_seed{self.seed}.json"
            output_path = Path(__file__).parent / "data" / filename

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize per-user data
        users_data = {}
        for user_id, user_set in dataset.users.items():
            users_data[str(user_id)] = {
                "members": [self._unit_to_dict(u) for u in user_set.members],
                "non_members": [self._unit_to_dict(u) for u in user_set.non_members],
            }

        if self.dataset_name == "locomo":
            split_unit = "locomo_session"
        elif self.dataset_name == "msc":
            split_unit = "msc_session_1_member_session_2_nonmember"
        else:
            split_unit = "dialogue_event_id"

        output_data = {
            "metadata": {
                "dataset_name": self.dataset_name,
                "perlt_types": [t.value for t in self.perlt_types],
                "seed": self.seed,
                "split_ratio": self.split_ratio,
                "split_unit": split_unit,
                "max_users": self.max_users,
            },
            "stats": dataset.stats(),
            "user_stats": dataset.user_stats(),
            "users": users_data,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        log.info("Saved dataset to %s", output_path)
        return output_path

    @staticmethod
    def _unit_to_dict(unit: MemoryUnit) -> dict:
        """Convert MemoryUnit to serializable dict."""
        return {
            "id": unit.id,
            "content": unit.content,
            "perlt_type": unit.perlt_type.value,
            "topic": unit.topic,
            "user_id": unit.user_id,
            "source_key": unit.source_key,
            "event_id": unit.event_id,
            "key_value": unit.key_value,
            "is_member": unit.is_member,
        }

    @classmethod
    def load(cls, input_path: str | Path) -> MemoryDataset:
        """Load a previously saved dataset.

        Args:
            input_path: Path to the saved JSON file

        Returns:
            MemoryDataset with loaded units
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {input_path}")

        with open(input_path, encoding="utf-8") as f:
            data = json.load(f)

        metadata = data.get("metadata", {})
        perlt_types = [PerltType(t) for t in metadata.get("perlt_types", [])]

        # Load per-user data
        users: dict[int, UserMemorySet] = {}
        for user_id_str, user_data in data.get("users", {}).items():
            user_id = int(user_id_str)
            members = [cls._dict_to_unit(d, is_member=True) for d in user_data.get("members", [])]
            non_members = [cls._dict_to_unit(d, is_member=False) for d in user_data.get("non_members", [])]
            users[user_id] = UserMemorySet(
                user_id=user_id,
                members=members,
                non_members=non_members,
            )

        dataset = MemoryDataset(
            users=users,
            dataset_name=metadata.get("dataset_name", ""),
            seed=metadata.get("seed", 42),
            perlt_types=perlt_types,
        )

        log.info("Loaded dataset from %s: %s", input_path, dataset.stats())
        return dataset

    @staticmethod
    def _dict_to_unit(d: dict, is_member: bool) -> MemoryUnit:
        """Convert dict back to MemoryUnit."""
        return MemoryUnit(
            id=d.get("id", ""),
            content=d.get("content", ""),
            perlt_type=PerltType(d.get("perlt_type", "profile")),
            topic=d.get("topic"),
            user_id=d.get("user_id", 0),
            source_key=d.get("source_key", ""),
            event_id=d.get("event_id") or MemoryExtractor._event_id_from_source_key(d.get("source_key", "")),
            key_value=d.get("key_value"),
            is_member=is_member,
        )

    # ── Utility methods ───────────────────────────────────────────────────

    def get_qa_pairs(self) -> list[dict]:
        """Load QA pairs from perltqa_en.json for evaluation."""
        qa_file = self.data_dir / "perltqa_en.json"
        if not qa_file.exists():
            return []

        with open(qa_file, encoding="utf-8") as f:
            return json.load(f)


# ── CLI for testing ───────────────────────────────────────────────────────

async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract dialogue memory units with event/session grouped splits"
    )
    parser.add_argument("--dataset", default="perltqa", help="Dataset name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-users",
        type=int,
        default=None,
        help="Limit users for testing; default uses all users except msc, which defaults to 50 pair-users",
    )
    parser.add_argument("--max-concurrency", type=int, default=32, help="Max parallel LLM calls")
    parser.add_argument("--output", default=None, help="Output JSON file")
    args = parser.parse_args()

    extractor = MemoryExtractor(
        dataset_name=args.dataset,
        seed=args.seed,
        max_users=args.max_users,
        max_concurrency=args.max_concurrency,
    )

    try:
        dataset = await extractor.extract()

        print("\n=== Dataset Stats ===")
        print(json.dumps(dataset.stats(), indent=2))

        print("\n=== Per-User Stats ===")
        for user_stats in dataset.user_stats():
            print(f"  User {user_stats['user_id']}: {user_stats['members']} members, {user_stats['non_members']} non-members")

        # Show sample from first user
        if dataset.user_ids:
            first_user = dataset.get_user(dataset.user_ids[0])
            print(f"\n=== Sample from User {first_user.user_id} ===")
            print("Members:")
            for unit in first_user.members[:2]:
                print(f"  [{unit.perlt_type.value}] {unit.content[:80]}...")
            print("Non-Members:")
            for unit in first_user.non_members[:2]:
                print(f"  [{unit.perlt_type.value}] {unit.content[:80]}...")

        # Save dataset
        if args.output:
            saved_path = extractor.save(dataset, args.output)
        else:
            saved_path = extractor.save(dataset)  # Auto-generate path
        print(f"\nSaved to {saved_path}")

    finally:
        await extractor.close()


if __name__ == "__main__":
    asyncio.run(main())

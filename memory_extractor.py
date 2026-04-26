"""Memory Extractor for CEA-MIA.

Extracts and splits memory units from perltqa-style datasets.
Supports LLM-based fact extraction from raw text.

Usage:
    extractor = MemoryExtractor(
        dataset_name="perltqa",
        perlt_types=[PerltType.PROFILE, PerltType.EVENT],
        seed=42,
    )
    dataset = await extractor.extract()
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


class MemoryExtractor:
    """Extract and split memory units from perltqa-style datasets."""

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
            perlt_types: Which types to extract (default: all)
            seed: Random seed for reproducible splits
            split_ratio: Fraction to use as members (default: 0.5)
            max_users: Limit number of users to process (for testing)
            max_concurrency: Max parallel LLM calls for event/dialogue extraction
            config: Config object for LLM settings
        """
        self.dataset_name = dataset_name
        self.perlt_types = perlt_types or list(PerltType)
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

        Split strategy: For each (user, type) group, split units 50/50.
        This ensures balanced splits within each user and type combination.
        """
        log.info("Loading dataset: %s", self.dataset_name)
        log.info("Types: %s", [t.value for t in self.perlt_types])
        log.info("Seed: %d, Split ratio: %.2f", self.seed, self.split_ratio)

        # Check if LLM is required but not configured
        llm_required_types = {PerltType.EVENT, PerltType.DIALOGUE}
        needs_llm = any(t in llm_required_types for t in self.perlt_types)
        if needs_llm and not self.config.api_base:
            log.warning("=" * 60)
            log.warning("LLM required for event/dialogue extraction but not configured!")
            log.warning("Set CEA_MI_API_BASE to your vLLM endpoint (e.g., http://localhost:8000/v1)")
            log.warning("Event and dialogue types will be skipped.")
            log.warning("=" * 60)

        # Load raw data
        raw_data = self._load_raw_data()
        if self.max_users:
            raw_data = raw_data[:self.max_users]
        log.info("Loaded %d users", len(raw_data))

        # Collect all extractable items grouped by source
        # Key: (user_idx, source_key), Value: raw data for that item
        all_items: list[tuple[int, str, PerltType, dict | str]] = []

        for user_idx, user_data in enumerate(raw_data):
            # Profile
            if PerltType.PROFILE in self.perlt_types:
                profile = user_data.get("profile", {})
                if profile:
                    all_items.append((user_idx, "profile", PerltType.PROFILE, profile))

            # Social relationships
            if PerltType.SOCIAL_RELATIONSHIP in self.perlt_types:
                social = user_data.get("social_relationship", {})
                for rel_key, rel_data in social.items():
                    all_items.append((user_idx, rel_key, PerltType.SOCIAL_RELATIONSHIP, rel_data))

            # Events
            if PerltType.EVENT in self.perlt_types:
                events = user_data.get("events", {})
                for event_key, event_data in events.items():
                    all_items.append((user_idx, event_key, PerltType.EVENT, event_data))

            # Dialogues
            if PerltType.DIALOGUE in self.perlt_types:
                dialogues = user_data.get("dialogues", {})
                for dial_key, dial_data in dialogues.items():
                    all_items.append((user_idx, dial_key, PerltType.DIALOGUE, dial_data))

        log.info("Found %d extractable items", len(all_items))

        # First extract ALL units (without membership label)
        all_units = await self._extract_from_items(
            all_items, is_member=False, max_concurrency=self.max_concurrency
        )
        log.info("Extracted %d total memory units", len(all_units))

        # Group units by user_id first, then by perlt_type within each user
        user_groups: dict[int, dict[PerltType, list[MemoryUnit]]] = {}
        for unit in all_units:
            if unit.user_id not in user_groups:
                user_groups[unit.user_id] = {}
            if unit.perlt_type not in user_groups[unit.user_id]:
                user_groups[unit.user_id][unit.perlt_type] = []
            user_groups[unit.user_id][unit.perlt_type].append(unit)

        log.info("Grouped into %d users", len(user_groups))

        # Build per-user memory sets with stratified splits
        user_memory_sets: dict[int, UserMemorySet] = {}

        for user_id in sorted(user_groups.keys()):
            user_members = []
            user_non_members = []

            for perlt_type, units in user_groups[user_id].items():
                # Shuffle within this (user, type) group
                self.rng.shuffle(units)
                split_idx = int(len(units) * self.split_ratio)

                # Handle edge case: if only 1 unit, randomly assign
                if len(units) == 1:
                    if self.rng.random() < self.split_ratio:
                        units[0].is_member = True
                        user_members.append(units[0])
                    else:
                        units[0].is_member = False
                        user_non_members.append(units[0])
                else:
                    # Split: first half members, second half non-members
                    for unit in units[:split_idx]:
                        unit.is_member = True
                        user_members.append(unit)
                    for unit in units[split_idx:]:
                        unit.is_member = False
                        user_non_members.append(unit)

                log.debug("  User %d, %s: %d members, %d non-members",
                          user_id, perlt_type.value,
                          len([u for u in units if u.is_member]),
                          len([u for u in units if not u.is_member]))

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
        """Load the raw perltmem JSON file."""
        mem_file = self.data_dir / "perltmem_en.json"
        if not mem_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {mem_file}")

        with open(mem_file, encoding="utf-8") as f:
            return json.load(f)

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
            # Auto-generate filename based on extraction params
            types_str = "_".join(t.value for t in self.perlt_types)
            filename = f"{self.dataset_name}_{types_str}_seed{self.seed}.json"
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

        output_data = {
            "metadata": {
                "dataset_name": self.dataset_name,
                "perlt_types": [t.value for t in self.perlt_types],
                "seed": self.seed,
                "split_ratio": self.split_ratio,
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

    parser = argparse.ArgumentParser(description="Extract memory units from perltqa dataset")
    parser.add_argument("--dataset", default="perltqa", help="Dataset name")
    parser.add_argument("--types", nargs="+", default=["dialogue"],
                        choices=["profile", "social_relationship", "event", "dialogue"],
                        help="Memory types to extract")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-users", type=int, default=5, help="Limit users for testing")
    parser.add_argument("--max-concurrency", type=int, default=10, help="Max parallel LLM calls")
    parser.add_argument("--output", default=None, help="Output JSON file")
    args = parser.parse_args()

    perlt_types = [PerltType(t) for t in args.types]

    extractor = MemoryExtractor(
        dataset_name=args.dataset,
        perlt_types=perlt_types,
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

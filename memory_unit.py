"""Simplified Memory Unit for CEA-MIA with perltqa support.

This module defines a streamlined memory representation that works with
the perltqa dataset format, replacing the more complex original models.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class PerltType(str, Enum):
    """Memory type categories matching perltqa dataset structure."""
    PROFILE = "profile"
    SOCIAL_RELATIONSHIP = "social_relationship"
    EVENT = "event"
    DIALOGUE = "dialogue"


@dataclass
class MemoryUnit:
    """A single memory unit extracted from perltqa data.

    This is a simplified representation focused on:
    - content: The actual memory text (what the agent "remembers")
    - perlt_type: Category of memory (profile/relation/event/dialogue)
    - perlt_reference: Reference key for ground truth matching with perltqa QA
    """
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    # Core fields
    content: str = ""                          # The memory text itself
    perlt_type: PerltType = PerltType.PROFILE  # Category
    topic: Optional[str] = None                # Shared topic/axis for probe pairing
    key_value: Optional[str] = None            # The core value/canonical answer to topic

    # Source tracking
    user_id: int = 0                           # Which user in perltmem (index)
    source_key: str = ""                       # Original key like "1_0_0" or "Gender"
    event_id: Optional[str] = None             # Dialogue/event group id used for grouped splits

    # Membership label (set during train/test split)
    is_member: bool = False                    # Whether this is in the agent's memory

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        status = "member" if self.is_member else "non-member"
        topic = self.topic or "none"
        key_value = self.key_value or "none"
        return (
            f"MemoryUnit({self.perlt_type.value}, topic={topic}, key_value={key_value}: "
            f"{self.content[:50]}... [{status}])"
        )


@dataclass
class MemoryUnitPair:
    """A memory unit paired with its counterfactual decoy."""
    original: MemoryUnit
    decoy: MemoryUnit


@dataclass
class UserMemorySet:
    """Memory units for a single user, split into members and non-members.

    This represents the attack scenario for one user:
    - members: memories that ARE in the agent's memory (ground truth positive)
    - non_members: memories that are NOT in the agent's memory (ground truth negative)
    """
    user_id: int
    members: list[MemoryUnit] = field(default_factory=list)
    non_members: list[MemoryUnit] = field(default_factory=list)

    @property
    def all_units(self) -> list[MemoryUnit]:
        return self.members + self.non_members

    def stats(self) -> dict:
        type_counts = {}
        for unit in self.all_units:
            t = unit.perlt_type.value
            type_counts[t] = type_counts.get(t, 0) + 1
        return {
            "user_id": self.user_id,
            "total": len(self.all_units),
            "members": len(self.members),
            "non_members": len(self.non_members),
            "by_type": type_counts,
        }


@dataclass
class MemoryDataset:
    """A dataset containing per-user memory splits.

    Each user has their own member/non-member split, enabling:
    1. Inject user X's members into agent memory
    2. Attack with user X's all_units
    3. Evaluate if attack distinguishes members from non-members
    """

    users: dict[int, UserMemorySet] = field(default_factory=dict)

    # Metadata
    dataset_name: str = ""
    seed: int = 42
    perlt_types: list[PerltType] = field(default_factory=list)

    def get_user(self, user_id: int) -> UserMemorySet:
        """Get memory set for a specific user."""
        if user_id not in self.users:
            raise KeyError(f"User {user_id} not found in dataset")
        return self.users[user_id]

    @property
    def all_members(self) -> list[MemoryUnit]:
        """All members across all users (for global stats)."""
        return [u for user_set in self.users.values() for u in user_set.members]

    @property
    def all_non_members(self) -> list[MemoryUnit]:
        """All non-members across all users (for global stats)."""
        return [u for user_set in self.users.values() for u in user_set.non_members]

    @property
    def all_units(self) -> list[MemoryUnit]:
        return self.all_members + self.all_non_members

    @property
    def user_ids(self) -> list[int]:
        return sorted(self.users.keys())

    def stats(self) -> dict:
        type_counts = {}
        for unit in self.all_units:
            t = unit.perlt_type.value
            type_counts[t] = type_counts.get(t, 0) + 1
        return {
            "num_users": len(self.users),
            "total": len(self.all_units),
            "members": len(self.all_members),
            "non_members": len(self.all_non_members),
            "by_type": type_counts,
            "seed": self.seed,
        }

    def user_stats(self) -> list[dict]:
        """Per-user statistics."""
        return [self.users[uid].stats() for uid in self.user_ids]

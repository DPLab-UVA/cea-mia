"""Data loader — create member/nonmember fact datasets for CEA-MI."""
from __future__ import annotations

import json
import random
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from config import DEFAULT_DATA_DIR
from models import Fact, DecoyPair


# Fact templates: (topic, template, possible_values, category)
FACT_TEMPLATES = [
    ("favorite_color", "The user's favorite color is {v}.", ["blue", "red", "green", "purple", "orange", "yellow", "black", "white", "pink", "teal"], "user_profile"),
    ("pet_name", "The user has a pet cat named {v}.", ["Mochi", "Luna", "Whiskers", "Shadow", "Cleo", "Bella", "Nala", "Simba", "Oreo", "Mittens"], "user_profile"),
    ("city", "The user lives in {v}.", ["Tokyo", "Berlin", "London", "Paris", "New York", "Seoul", "Sydney", "Toronto", "Singapore", "Amsterdam"], "user_profile"),
    ("job_title", "The user works as a {v}.", ["data scientist", "backend engineer", "product manager", "ML researcher", "security analyst", "DevOps engineer", "frontend developer", "systems architect", "UX designer", "tech lead"], "user_profile"),
    ("coffee", "The user's preferred coffee drink is {v}.", ["espresso", "latte", "cappuccino", "americano", "cold brew", "flat white", "mocha", "pour over", "matcha latte", "black coffee"], "user_profile"),
    ("language", "The user's primary programming language is {v}.", ["Python", "Rust", "Go", "TypeScript", "Java", "C++", "Kotlin", "Swift", "Ruby", "Scala"], "user_profile"),
    ("editor", "The user's preferred code editor is {v}.", ["VS Code", "Neovim", "IntelliJ", "Emacs", "Sublime Text", "Zed", "Helix", "PyCharm", "Cursor", "Vim"], "user_profile"),
    ("os", "The user's daily operating system is {v}.", ["macOS", "Ubuntu", "Arch Linux", "Windows 11", "Fedora", "NixOS", "Debian", "ChromeOS", "FreeBSD", "Pop!_OS"], "user_profile"),
    ("hobby", "The user's main hobby is {v}.", ["rock climbing", "photography", "cooking", "chess", "gardening", "running", "painting", "reading", "gaming", "cycling"], "user_profile"),
    ("birthday_month", "The user's birthday is in {v}.", ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October"], "user_profile"),
    ("partner_name", "The user's partner is named {v}.", ["Alex", "Jamie", "Sam", "Taylor", "Jordan", "Casey", "Morgan", "Riley", "Quinn", "Avery"], "user_profile"),
    ("university", "The user graduated from {v}.", ["MIT", "Stanford", "Tsinghua", "ETH Zurich", "Cambridge", "Peking University", "Georgia Tech", "CMU", "UC Berkeley", "Oxford"], "user_profile"),
    ("music_genre", "The user's favorite music genre is {v}.", ["jazz", "classical", "indie rock", "hip hop", "electronic", "folk", "R&B", "metal", "lo-fi", "blues"], "user_profile"),
    ("food", "The user's favorite cuisine is {v}.", ["Japanese ramen", "Italian pasta", "Thai curry", "Mexican tacos", "Indian biryani", "Korean BBQ", "French pastry", "Vietnamese pho", "Ethiopian injera", "Sichuan hotpot"], "user_profile"),
    ("framework", "The project uses {v} as the main ML framework.", ["PyTorch", "TensorFlow", "JAX", "scikit-learn", "Hugging Face Transformers", "LightGBM", "XGBoost", "Keras", "FastAI", "MLX"], "technical"),
    ("database", "The team's primary database is {v}.", ["PostgreSQL", "MongoDB", "Redis", "DynamoDB", "Cassandra", "ClickHouse", "SQLite", "CockroachDB", "Neo4j", "TimescaleDB"], "technical"),
    ("cloud", "The project is deployed on {v}.", ["AWS", "GCP", "Azure", "Hetzner", "DigitalOcean", "Fly.io", "Cloudflare", "Vercel", "Railway", "Linode"], "technical"),
    ("gpu", "The training cluster uses {v} GPUs.", ["A100", "H100", "RTX 4090", "V100", "A6000", "RTX 3090", "L40S", "H200", "TPU v5", "MI300X"], "technical"),
    ("batch_size", "The default training batch size is {v}.", ["32", "64", "128", "256", "512", "1024", "16", "48", "96", "2048"], "technical"),
    ("learning_rate", "The optimizer learning rate is {v}.", ["1e-4", "3e-4", "1e-3", "5e-5", "2e-4", "1e-5", "5e-4", "3e-5", "7e-4", "2e-3"], "technical"),
]


class DataLoader:
    """Create and manage member/nonmember fact datasets."""

    def __init__(self, seed: int = 42, data_dir: Optional[Path] = None):
        self.rng = random.Random(seed)
        self.seed = seed
        self.data_dir = Path(data_dir).expanduser() if data_dir else DEFAULT_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def generate_facts(self, num_member: int = 50, num_nonmember: int = 50) -> tuple[list[Fact], list[Fact]]:
        """Generate member and nonmember facts from templates.

        For each template, randomly assign one value as member and another as nonmember.
        This ensures member and nonmember come from the SAME distribution (IID).
        """
        templates = list(FACT_TEMPLATES)
        self.rng.shuffle(templates)

        members = []
        nonmembers = []

        # We may need to reuse templates if num > len(templates)
        template_pool = templates * ((max(num_member, num_nonmember) // len(templates)) + 2)

        for i in range(max(num_member, num_nonmember)):
            topic, template, values, category = template_pool[i % len(templates)]
            # Pick two distinct values
            vals = self.rng.sample(values, min(2, len(values)))

            if i < num_member:
                member_val = vals[0]
                members.append(Fact(
                    content=template.format(v=member_val),
                    topic=f"{topic}_{i}",
                    key_value=member_val,
                    category=category,
                    is_member=True,
                ))

            if i < num_nonmember:
                nonmember_val = vals[1] if len(vals) > 1 else vals[0]
                nonmembers.append(Fact(
                    content=template.format(v=nonmember_val),
                    topic=f"{topic}_{i}",
                    key_value=nonmember_val,
                    category=category,
                    is_member=False,
                ))

        return members, nonmembers

    def build_decoy_pairs(self, facts: list[Fact], all_facts: list[Fact]) -> list[DecoyPair]:
        """For each fact, construct a counterfactual decoy (f-) and optional hard negative."""
        pairs = []
        for fact in facts:
            # Find the template for this fact
            base_topic = fact.topic.rsplit("_", 1)[0]
            matching_template = None
            for topic, template, values, category in FACT_TEMPLATES:
                if topic == base_topic:
                    matching_template = (topic, template, values, category)
                    break

            if matching_template is None:
                continue

            _, template, values, _ = matching_template
            # Pick a different value for decoy
            other_vals = [v for v in values if v != fact.key_value]
            if not other_vals:
                continue
            decoy_val = self.rng.choice(other_vals)

            decoy = Fact(
                content=template.format(v=decoy_val),
                topic=fact.topic + "_decoy",
                key_value=decoy_val,
                category=fact.category,
                is_member=False,
            )

            # Hard negative: same topic category but different fact
            hard_neg = None
            same_cat = [f for f in all_facts if f.category == fact.category and f.id != fact.id]
            if same_cat:
                hard_neg = self.rng.choice(same_cat)

            pairs.append(DecoyPair(fact=fact, decoy=decoy, hard_negative=hard_neg))

        return pairs

    def save_dataset(self, members: list[Fact], nonmembers: list[Fact], pairs: list[DecoyPair]):
        """Save the dataset to disk."""
        def fact_to_dict(f: Fact) -> dict:
            return {
                "id": f.id, "content": f.content, "topic": f.topic,
                "key_value": f.key_value, "category": f.category,
                "is_member": f.is_member, "memory_id": f.memory_id,
            }

        data = {
            "seed": self.seed,
            "members": [fact_to_dict(f) for f in members],
            "nonmembers": [fact_to_dict(f) for f in nonmembers],
            "decoy_pairs": [
                {
                    "fact": fact_to_dict(p.fact),
                    "decoy": fact_to_dict(p.decoy),
                    "hard_negative": fact_to_dict(p.hard_negative) if p.hard_negative else None,
                }
                for p in pairs
            ],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        out_path = self.data_dir / f"dataset_seed{self.seed}.json"
        out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        return out_path

    def load_dataset(self, seed: int = None) -> dict:
        """Load a previously saved dataset."""
        s = seed or self.seed
        path = self.data_dir / f"dataset_seed{s}.json"
        return json.loads(path.read_text())

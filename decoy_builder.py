"""Decoy builder — construct counterfactual facts for contrastive probing."""
from __future__ import annotations

import asyncio
from typing import Optional

import httpx

from models import Fact, DecoyPair


class DecoyBuilder:
    """Construct high-quality decoy facts using LLM assistance.

    For each candidate fact f, builds f- that:
    - Has the same topic, structure, and length
    - Replaces the key value with a plausible alternative
    - Is equally "natural" and "answerable" (critical for avoiding bias)
    """

    def __init__(
        self,
        api_base: str = "http://cheetah04:8000/v1",
        api_key: str = "token-vllm",
        model: str = "/bigtemp/trv3px/model_checkpoints/models--Qwen--Qwen2.5-72B-Instruct/snapshots/495f39366efef23836d0cfae4fbe635880d2be31",
    ):
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.client = httpx.AsyncClient(timeout=60)

    async def build_llm_decoy(self, fact: Fact) -> Fact:
        """Use LLM to generate a high-quality counterfactual decoy."""
        prompt = f"""Given this personal fact about a user:
"{fact.content}"

Generate ONE alternative version that:
1. Has the SAME structure and length
2. Replaces the key detail with a different but equally plausible value
3. Sounds natural and believable

Reply with ONLY the alternative fact, nothing else."""

        resp = await self.client.post(
            f"{self.api_base}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.8,
                "max_tokens": 128,
            },
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip().strip('"')

        return Fact(
            content=content,
            topic=fact.topic + "_llm_decoy",
            key_value="[llm_generated]",
            category=fact.category,
            is_member=False,
        )

    async def build_hard_negative(self, fact: Fact) -> Fact:
        """Generate a hard negative: same topic domain but entirely different fact."""
        prompt = f"""Given this fact about a user:
"{fact.content}"

Generate a DIFFERENT fact about the same general topic domain (e.g., if it's about food preference, generate a different food-related fact). The fact should:
1. Be about the same general domain
2. Contain DIFFERENT specific information
3. Be plausible as a real user fact

Reply with ONLY the fact, nothing else."""

        resp = await self.client.post(
            f"{self.api_base}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.9,
                "max_tokens": 128,
            },
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip().strip('"')

        return Fact(
            content=content,
            topic=fact.topic + "_hard_neg",
            key_value="[llm_generated]",
            category=fact.category,
            is_member=False,
        )

    async def build_decoy_pair(self, fact: Fact) -> DecoyPair:
        """Build a full decoy pair with both counterfactual and hard negative."""
        decoy, hard_neg = await asyncio.gather(
            self.build_llm_decoy(fact),
            self.build_hard_negative(fact),
        )
        return DecoyPair(fact=fact, decoy=decoy, hard_negative=hard_neg)

    async def close(self):
        await self.client.aclose()

"""Semantic paraphrasing for test-time candidate memory units."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import replace
from pathlib import Path
from typing import Optional

import httpx

from config import DEFAULT_API_BASE, DEFAULT_API_KEY, DEFAULT_MODEL
from http_utils import post_with_retry
from json_utils import JsonObjectError, extract_json_object_text, loads_json_object
from memory_attack_utils import UserAttackSet
from memory_unit import MemoryUnit


log = logging.getLogger(__name__)


SEMANTIC_PARAPHRASE_PROMPT = """\
Rewrite the candidate memory statement using different wording and sentence
structure while preserving the same meaning.

Candidate memory statement:
{content}

Requirements:
1. Preserve every named entity, relationship, attribute, time/date, place,
   number, and other factual detail.
2. Do not add, remove, generalize, or make any fact more specific.
3. Do not change the speaker, subject, polarity, chronology, or certainty.
4. The result must be a natural standalone declarative memory statement, not
   an explanation of the rewrite and not a direct question.
5. Return exactly one paraphrase.
6. If the original statement contains quoted speech, prefer indirect speech
   without quotation marks inside the paraphrase.
7. If the original says someone asks, says, wonders, or wants to know something,
   preserve that as a declarative statement about the asking/wondering. Do not
   turn it into the question they asked.

Return ONLY valid JSON in this exact shape:
{{
  "paraphrase": "rewritten memory statement"
}}
"""


class MemoryParaphraser:
    """Create semantic paraphrases of candidate memories before testing."""

    def __init__(
        self,
        api_base: str = DEFAULT_API_BASE,
        api_key: str = DEFAULT_API_KEY,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.3,
        concurrency: int = 20,
        mode: str = "semantic",
        cache_path: Optional[Path] = None,
        max_retries: int = 2,
    ):
        if mode != "semantic":
            raise ValueError(f"Unsupported paraphrase mode: {mode}")
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.mode = mode
        self.max_retries = max(0, int(max_retries))
        self.semaphore = asyncio.Semaphore(max(1, int(concurrency)))
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, str] = self._load_cache()
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    def _load_cache(self) -> dict[str, str]:
        if not self.cache_path or not self.cache_path.exists():
            return {}
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Could not read paraphrase cache %s: %s", self.cache_path, exc)
            return {}

        if isinstance(payload, dict):
            entries = payload.get("entries", payload)
            if isinstance(entries, dict):
                return {
                    str(key): str(value)
                    for key, value in entries.items()
                    if isinstance(value, str) and value.strip()
                }
        log.warning("Ignoring invalid paraphrase cache format: %s", self.cache_path)
        return {}

    def save_cache(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mode": self.mode,
            "model": self.model,
            "entries": self.cache,
        }
        self.cache_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    async def close(self) -> None:
        self.save_cache()
        if self._client:
            await self._client.aclose()
            self._client = None

    def _cache_key(self, content: str) -> str:
        raw = f"{self.mode}\n{self.model}\n{content}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _looks_like_direct_question(text: str) -> bool:
        normalized = (text or "").strip().lower()
        if "?" not in normalized:
            return False
        question_starts = (
            "is ",
            "are ",
            "am ",
            "was ",
            "were ",
            "do ",
            "does ",
            "did ",
            "can ",
            "could ",
            "will ",
            "would ",
            "should ",
            "has ",
            "have ",
            "had ",
            "what ",
            "where ",
            "when ",
            "why ",
            "how ",
            "who ",
            "which ",
        )
        return normalized.startswith(question_starts)

    @classmethod
    def _validate_paraphrase(cls, paraphrase: str) -> None:
        if cls._looks_like_direct_question(paraphrase):
            raise JsonObjectError(
                "paraphrase must be a declarative memory statement, not a direct question"
            )

    async def _call_llm(self, prompt: str, temperature: float) -> str:
        resp = await post_with_retry(
            self.client,
            f"{self.api_base}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": 512,
            },
        )
        return resp.json()["choices"][0]["message"]["content"].strip()

    @staticmethod
    def _normalize_paraphrase_text(text: str) -> str:
        text = (text or "").strip()
        if text.startswith("```json"):
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif text.startswith("```"):
            text = text.split("```", 1)[1].split("```", 1)[0].strip()

        # If malformed JSON dropped the opening quote of quoted speech, keep
        # the recovered statement readable instead of failing the whole sample.
        if text.count('"') % 2 == 1 and not text.startswith('"'):
            quote_idx = text.find('"')
            tail = text[quote_idx + 1 :].lstrip()
            if tail.lower().startswith(("asked ", "said ", "inquired ", "replied ")):
                text = f'"{text}'
            else:
                text = re.sub(r'([?.!])"\s+', r"\1 ", text, count=1)
        return text.strip()

    @staticmethod
    def _parse_loose_paraphrase(raw: str) -> str:
        """Recover a paraphrase from common malformed JSON string outputs."""
        text = extract_json_object_text(raw)
        marker = '"paraphrase"'
        idx = text.find(marker)
        if idx < 0:
            raise JsonObjectError("Missing paraphrase field")

        colon = text.find(":", idx + len(marker))
        if colon < 0:
            raise JsonObjectError("Missing ':' after paraphrase field")

        value = text[colon + 1 :].strip()
        if value.endswith("}"):
            value = value[:-1].strip()
        if value.endswith(","):
            value = value[:-1].strip()
        if value.startswith('"'):
            value = value[1:].strip()
        if value.endswith('"'):
            value = value[:-1].strip()

        value = value.replace("\\n", " ").replace("\n", " ").strip()
        value = MemoryParaphraser._normalize_paraphrase_text(value)
        if value:
            return value
        raise JsonObjectError("paraphrase must be a non-empty string")

    @staticmethod
    def _parse_plain_text_paraphrase(raw: str) -> str:
        text = MemoryParaphraser._normalize_paraphrase_text(raw)
        if text.startswith("{") and text.endswith("}"):
            raise JsonObjectError("Malformed JSON object did not contain a usable paraphrase")
        text = text.strip()
        if text.startswith('"') and text.endswith('"') and text.count('"') == 2:
            text = text[1:-1].strip()
        if text:
            return text
        raise JsonObjectError("paraphrase must be a non-empty string")

    @classmethod
    def _parse_paraphrase(cls, raw: str) -> str:
        try:
            payload = loads_json_object(raw, required_keys=("paraphrase",))
            paraphrase = cls._normalize_paraphrase_text(
                str(payload.get("paraphrase") or "")
            )
            if not paraphrase:
                raise JsonObjectError("paraphrase must be a non-empty string")
            return paraphrase
        except JsonObjectError as json_error:
            try:
                return cls._parse_loose_paraphrase(raw)
            except JsonObjectError:
                try:
                    return cls._parse_plain_text_paraphrase(raw)
                except JsonObjectError as plain_error:
                    raise JsonObjectError(
                        f"{json_error}; loose/plain fallback failed: {plain_error}"
                    ) from plain_error

    async def paraphrase_text(self, content: str) -> tuple[str, bool, Optional[str]]:
        content = (content or "").strip()
        if not content:
            return content, True, None

        cache_key = self._cache_key(content)
        cached = self.cache.get(cache_key)
        if cached:
            return cached, True, None

        prompt = SEMANTIC_PARAPHRASE_PROMPT.format(content=content)
        attempts = max(1, self.max_retries + 1)
        last_error: Optional[Exception] = None

        for attempt in range(1, attempts + 1):
            raw = await self._call_llm(
                prompt,
                self.temperature if attempt == 1 else 0.0,
            )
            try:
                paraphrase = self._parse_paraphrase(raw)
                self._validate_paraphrase(paraphrase)
                self.cache[cache_key] = paraphrase
                return paraphrase, False, None
            except JsonObjectError as exc:
                last_error = exc
                log.warning(
                    "Invalid paraphrase JSON (attempt %d/%d): %s | raw=%r",
                    attempt,
                    attempts,
                    exc,
                    raw[:500].replace("\n", "\\n"),
                )

        failure_reason = str(last_error) if last_error else "unknown paraphrase failure"
        log.warning(
            "Using original memory text after paraphrase failed: %s",
            failure_reason,
        )
        return content, False, failure_reason

    async def paraphrase_unit(
        self,
        unit: MemoryUnit,
        split: str,
    ) -> tuple[MemoryUnit, dict]:
        async with self.semaphore:
            paraphrased_content, from_cache, failure_reason = await self.paraphrase_text(
                unit.content
            )

        paraphrased_unit = replace(unit, content=paraphrased_content)
        record = {
            "fact_id": unit.id,
            "user_id": unit.user_id,
            "split": split,
            "is_member": unit.is_member,
            "source_key": unit.source_key,
            "topic": unit.topic,
            "key_value": unit.key_value,
            "paraphrase_mode": self.mode,
            "from_cache": from_cache,
            "paraphrase_failed": failure_reason is not None,
            "failure_reason": failure_reason,
            "original_content": unit.content,
            "paraphrased_content": paraphrased_content,
        }
        return paraphrased_unit, record

    async def paraphrase_attack_sets(
        self,
        user_attack_sets: list[UserAttackSet],
    ) -> tuple[list[UserAttackSet], list[dict]]:
        paraphrased_sets: list[UserAttackSet] = []
        records: list[dict] = []

        for user_attack_set in user_attack_sets:
            member_results = await asyncio.gather(*[
                self.paraphrase_unit(unit, "member")
                for unit in user_attack_set.members
            ])
            nonmember_results = await asyncio.gather(*[
                self.paraphrase_unit(unit, "non_member")
                for unit in user_attack_set.non_members
            ])

            members = [unit for unit, _ in member_results]
            non_members = [unit for unit, _ in nonmember_results]
            records.extend(record for _, record in member_results)
            records.extend(record for _, record in nonmember_results)
            paraphrased_sets.append(
                UserAttackSet(
                    user_id=user_attack_set.user_id,
                    members=members,
                    non_members=non_members,
                )
            )

        return paraphrased_sets, records

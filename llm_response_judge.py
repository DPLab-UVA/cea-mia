"""LLM-based response judging for natural attack probe answers."""
from __future__ import annotations

import json
import logging
from typing import Optional

import httpx

from config import DEFAULT_API_BASE, DEFAULT_API_KEY, DEFAULT_MODEL
from http_utils import post_with_retry
from models import ProbeType


log = logging.getLogger(__name__)


# Used by MRMMIA/natural_attack and multi-probe baselines when response_scorer="llm".
# Scores an agent response against a probe-level topic/key_value.
JUDGE_PROMPT = """\
You are a strict evaluator for a memory membership probing experiment.

Judge the AGENT RESPONSE using two checks:
1. Does it support the specified KEY_VALUE?
2. Does its stated reason/source fully align with the CANDIDATE STATEMENT?

Probe type: {probe_type}
Topic: {topic}
Key value: {key_value}

CANDIDATE STATEMENT:
{candidate_statement}

QUESTION:
{question}

AGENT RESPONSE:
{response}

Important rules:
- **Question leakage**: The QUESTION is context only; values only in the
  QUESTION are not evidence.
- **Key gate**: If the response does not support the key_value, is unknown,
  contradicts the statement, or gives a different value, score 0.0.
- **Reason gate**: If the key_value is supported but the reason/source is
  missing, incomplete, inferred, uncertain, or loosely related, score at most
  0.33.
- **High score gate**: Give 1.0 or 0.67 only when the reason/source matches
  the candidate statement without missing important information.
- **Fixed scale**: Use only these four scores: 1.0, 0.67, 0.33, 0.0.

Scoring rubric for this probe type:
{rubric}

Return ONLY valid JSON in this exact shape:
{{
  "score": 1.0
}}
"""

NO_REASON_RECALL_JUDGE_PROMPT = """\
You are a strict evaluator for a memory membership probing experiment.

Judge whether the AGENT RESPONSE supports the specified KEY_VALUE for the
given direct-recall QUESTION. Do not require the response to explain how it
knows the answer.

Topic: {topic}
Key value: {key_value}

CANDIDATE STATEMENT:
{candidate_statement}

QUESTION:
{question}

AGENT RESPONSE:
{response}

Important rules:
- **Question leakage**: The QUESTION is context only; values only in the
  QUESTION are not evidence.
- **Key support only**: Score the answer to the direct recall question. Do not
  require a source, reason, provenance, or memory citation.
- **Unknown is zero**: Treat "I don't know", "I don't remember", "no information",
  refusals, and similar responses as 0.0.
- **Wrong value is zero**: If the response gives a different value for the same
  topic, contradicts the candidate statement, or only gives broad related
  information, score 0.0.
- **Fixed scale**: Use only these four scores: 1.0, 0.67, 0.33, 0.0.

Scoring rubric:
- 1.0 full_match: The response clearly gives the same specific key value.
- 0.67 near_match: The response gives a true synonym, paraphrase, or near match of the key value.
- 0.33 partial_match: The response gives a weak but specific clue toward the key value, or an incomplete version of it.
- 0.0 no_match: The key value is absent or wrong, merely present in the question, contradicted, replaced by another value, unknown, or only broadly related.

Return ONLY valid JSON in this exact shape:
{{
  "score": 1.0
}}
"""

# Used by whitebox multi-probe baselines when recalled memory snippets are
# available. Scores recalled memory text similarity against topic/key_value.
MEMORY_JUDGE_PROMPT = """\
You are a strict evaluator for a white-box memory membership experiment.

Judge how closely the RECALLED MEMORY text matches the CANDIDATE STATEMENT
on the specified TOPIC and KEY_VALUE.

Topic: {topic}
Key value: {key_value}

CANDIDATE STATEMENT:
{candidate_statement}

QUESTION:
{question}

RECALLED MEMORY:
{memory_content}

Important rules:
- **Question leakage**: The QUESTION is context only. A value mentioned only in the QUESTION is not
  evidence.
- **Stored text**: Treat the recalled memory as stored memory text, not as an assistant answer.
- **Contradiction is zero**: Treat explicit contradiction of the candidate statement as 0.0.
- **Similarity, not support**: Score direct text/information match only. Do not give high credit for inferred, entailed, or broadly compatible memories.
- **Wrong value is zero**: If the recalled memory states a different value on the same topic, score 0.0 even if it is plausible or related.
- **Fixed scale**: Use only these four scores: 1.0, 0.67, 0.33, 0.0.

Scoring rubric:
{rubric}

Return ONLY valid JSON in this exact shape:
{{
  "score": 1.0
}}
"""

# Used by Naive Probe and Reference baselines. Scores an agent response against
# the full candidate memory statement, not just a probe-level key_value.
STATEMENT_JUDGE_PROMPT = """\
You are a strict evaluator for a memory membership probing experiment.

Judge whether the AGENT RESPONSE supports the CANDIDATE STATEMENT.
The QUESTION is context only: information mentioned only in the QUESTION is
not evidence unless the RESPONSE independently affirms or uses it as true.

CANDIDATE STATEMENT:
{candidate_statement}

QUESTION:
{question}

AGENT RESPONSE:
{response}

Important rules:
- **Unknown is zero**: Treat "I don't know", "I don't remember", "no information", and similar responses as 0.0.
- **Contradiction is zero**: Treat explicit contradiction of the candidate statement as 0.0.
- **Specific support matters**: Broadly related responses are not enough unless they support the specific statement.
- **Question leakage is not evidence**: Do not give credit for facts that appear only in the question.
- **Fixed scale**: Use only these four scores: 1.0, 0.67, 0.33, 0.0.

Scoring rubric:
- 1.0 full_match: The response clearly supports the candidate statement with the same specific details.
- 0.67 near_match: The response supports the same statement with a true synonym, paraphrase, or minor missing non-critical detail.
- 0.33 partial_match: The response gives a weak but specific clue toward the candidate statement.
- 0.0 no_match: The response is absent, unknown, merely related, contradicted, only repeats question content, or supports a different statement.

Return ONLY valid JSON in this exact shape:
{{
  "score": 1.0
}}
"""

# Used by MRMMIA/natural_attack, Naive Probe, Reference, and MultiJudge whitebox
# paths when recalled memory snippets are available. Scores recalled memory
# text similarity against the full statement.
MEMORY_STATEMENT_JUDGE_PROMPT = """\
You are a strict evaluator for a white-box memory membership experiment. 

Judge the text and information similarity between the RECALLED MEMORY and CANDIDATE STATEMENT.
The QUESTION is context only, not evidence.

CANDIDATE STATEMENT:
{candidate_statement}

QUESTION:
{question}

RECALLED MEMORY:
{memory_content}

Important rules:
- **Stored text**: Treat the recalled memory as stored memory text, not as an assistant answer.
- **Contradiction is zero**: Treat explicit contradiction of the candidate statement as 0.0.
- **Text match matters**: 1.0 requires an almost verbatim match of the candidate statement, not merely support.
- **Information match matters**: 0.67 requires the same concrete information and all critical details, with only minor lexical or syntactic differences.
- **No inference credit**: Broadly related, entailed, or compatible memories are not enough for 0.67 or 1.0.
- **Question leakage is not evidence**: Do not give credit for facts that appear only in the question.
- **Fixed scale**: Use only these four scores: 1.0, 0.67, 0.33, 0.0.

Scoring rubric:
- 1.0 text_match: The recalled memory is almost identical to the candidate statement, preserving the same subject, relation, value, critical details, key wording, and statement framing.
- 0.67 information_match: The recalled memory states the same concrete information with all critical details, but has only minor lexical changes, synonyms, word-order changes, or small grammatical differences.
- 0.33 partial_or_related_match: The recalled memory shares a specific entity, event, or fragment with the candidate, but omits or changes a critical detail, requires inference, or is only a loose paraphrase.
- 0.0 no_match: The recalled memory is absent, unrelated, only broadly compatible, contradicted, repeats the question without evidence, or states a different claim/value.

Return ONLY valid JSON in this exact shape:
{{
  "score": 1.0
}}
"""

# old rubric
# Scoring rubric:
# - 1.0 full_match: The recalled memory clearly supports the candidate statement with the same specific details and key words.
# - 0.67 near_match: The recalled memory supports the same statement with a true synonym, paraphrase, or minor missing non-critical detail.
# - 0.33 partial_match: The recalled memory gives a weak but specific clue toward the candidate statement.
# - 0.0 no_match: The memory is absent, merely related, contradicted, only repeats question content, or supports a different statement.

# Rubrics inserted into response and memory judges.
RECALL_RUBRIC = """\
- 1.0 full_match: The response gives the same specific key value and its stated reason/source directly matches the candidate statement, such as citing the same remembered fact, prior conversation, or stored information.
- 0.67 near_match: The response gives a true synonym/paraphrase of the specific key value and its stated reason/source is a close paraphrase or near match of the candidate statement.
- 0.33 partial_match: The response gives the exact or near-match key value, but the stated reason/source is missing, incomplete, only loosely related to the candidate statement, inferred from general context/world knowledge, provided with extra unrelated statement, or expressed with uncertainty/speculation.
- 0.0 no_match: The key value is absent or wrong, merely present in the question, contradicted, replaced by another value, supported only by unrelated/broad/compatible information, or the response is unknown/refusal/no information."""

MEMORY_SUPPORT_RUBRIC = """\
- 1.0 text_match: The recalled memory is almost identical to the candidate statement on the specified topic/key value, preserving the same subject, relation, value, critical details, key wording, and framing.
- 0.67 information_match: The recalled memory states the same concrete information and all critical topic/key-value details, but has only minor lexical changes, synonyms, word-order changes, or small grammatical differences.
- 0.33 partial_or_related_match: The recalled memory shares a specific entity, event, key value, or fragment, but omits or changes a critical detail, requires inference, or is only a loose paraphrase.
- 0.0 no_match: The recalled memory is absent, unrelated, only broadly compatible, contradicted, only repeats question content, or states a different claim/value."""

PROVENANCE_RUBRIC = """\
- 1.0 full_match: The response explicitly supports the same specific key value and gives a clear memory/source basis, such as "you told me", "you mentioned", "I remember", or similar.
- 0.67 near_match: The response supports the same specific key value, but the memory/source basis is weak or the value is a true synonym/paraphrase.
- 0.33 partial_match: The response gives a specific clue to the same key value, or gives the exact key value without a clear provenance basis.
- 0.0 no_match: The key value is absent, only broad/entailed/compatible, merely present in the question, contradicted, replaced by another value, or unknown."""

CONFIRMATION_RUBRIC = """\
- 1.0 full_match: The response explicitly confirms the candidate statement and independently supports the same specific key value.
- 0.67 near_match: The response confirms the statement with a true synonym/paraphrase of the same specific key value.
- 0.33 partial_match: The response weakly supports the same key value but is hedged or incomplete.
- 0.0 no_match: The response merely agrees with the question, confirms only the topic, supports a different/broader/compatible value, denies the statement, is unrelated, or unknown."""

class LLMResponseJudge:
    """Score probe responses with an LLM judge."""

    def __init__(
        self,
        api_base: str = DEFAULT_API_BASE,
        api_key: str = DEFAULT_API_KEY,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
    ):
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
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

    @staticmethod
    def _extract_json_text(raw: str) -> str:
        text = (raw or "").strip()
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0]
        return text.strip()

    @staticmethod
    def _normalize_score(value) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        allowed = (0.0, 0.33, 0.67, 1.0)
        return min(allowed, key=lambda x: abs(x - score))

    @staticmethod
    def _rubric_for(probe_type: ProbeType) -> str:
        if probe_type == ProbeType.PROVENANCE:
            return PROVENANCE_RUBRIC
        if probe_type == ProbeType.CONFIRMATION:
            return CONFIRMATION_RUBRIC
        return RECALL_RUBRIC

    async def judge(
        self,
        candidate_statement: str,
        topic: str,
        key_value: str,
        question: str,
        response: str,
        probe_type: ProbeType,
    ) -> float:
        prompt = JUDGE_PROMPT.format(
            probe_type=probe_type.value,
            topic=topic or "<none>",
            key_value=key_value or "<none>",
            candidate_statement=candidate_statement,
            question=question,
            response=response,
            rubric=self._rubric_for(probe_type),
        )
        resp = await post_with_retry(
            self.client,
            f"{self.api_base}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": 256,
            },
        )
        raw = resp.json()["choices"][0]["message"]["content"]
        try:
            payload = json.loads(self._extract_json_text(raw))
        except json.JSONDecodeError:
            log.warning("LLM judge returned non-JSON response: %s", raw)
            return 0.0

        return self._normalize_score(payload.get("score"))

    async def judge_recall_no_reason(
        self,
        candidate_statement: str,
        topic: str,
        key_value: str,
        question: str,
        response: str,
    ) -> float:
        prompt = NO_REASON_RECALL_JUDGE_PROMPT.format(
            topic=topic or "<none>",
            key_value=key_value or "<none>",
            candidate_statement=candidate_statement,
            question=question,
            response=response,
        )
        resp = await post_with_retry(
            self.client,
            f"{self.api_base}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": 256,
            },
        )
        raw = resp.json()["choices"][0]["message"]["content"]
        try:
            payload = json.loads(self._extract_json_text(raw))
        except json.JSONDecodeError:
            log.warning("LLM no-reason recall judge returned non-JSON response: %s", raw)
            return 0.0

        return self._normalize_score(payload.get("score"))

    async def judge_statement(
        self,
        candidate_statement: str,
        question: str,
        response: str,
    ) -> float:
        prompt = STATEMENT_JUDGE_PROMPT.format(
            candidate_statement=candidate_statement,
            question=question,
            response=response,
        )
        resp = await post_with_retry(
            self.client,
            f"{self.api_base}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": 256,
            },
        )
        raw = resp.json()["choices"][0]["message"]["content"]
        try:
            payload = json.loads(self._extract_json_text(raw))
        except json.JSONDecodeError:
            log.warning("LLM statement judge returned non-JSON response: %s", raw)
            return 0.0

        return self._normalize_score(payload.get("score"))

    async def judge_memory_statement(
        self,
        candidate_statement: str,
        question: str,
        memory_content: str,
    ) -> float:
        prompt = MEMORY_STATEMENT_JUDGE_PROMPT.format(
            candidate_statement=candidate_statement,
            question=question,
            memory_content=memory_content,
        )
        resp = await post_with_retry(
            self.client,
            f"{self.api_base}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": 256,
            },
        )
        raw = resp.json()["choices"][0]["message"]["content"]
        try:
            payload = json.loads(self._extract_json_text(raw))
        except json.JSONDecodeError:
            log.warning("LLM memory statement judge returned non-JSON response: %s", raw)
            return 0.0

        return self._normalize_score(payload.get("score"))

    async def judge_memory(
        self,
        candidate_statement: str,
        topic: str,
        key_value: str,
        question: str,
        memory_content: str,
    ) -> float:
        prompt = MEMORY_JUDGE_PROMPT.format(
            topic=topic or "<none>",
            key_value=key_value or "<none>",
            candidate_statement=candidate_statement,
            question=question,
            memory_content=memory_content,
            rubric=MEMORY_SUPPORT_RUBRIC,
        )
        resp = await post_with_retry(
            self.client,
            f"{self.api_base}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": 256,
            },
        )
        raw = resp.json()["choices"][0]["message"]["content"]
        try:
            payload = json.loads(self._extract_json_text(raw))
        except json.JSONDecodeError:
            log.warning("LLM memory judge returned non-JSON response: %s", raw)
            return 0.0

        return self._normalize_score(payload.get("score"))

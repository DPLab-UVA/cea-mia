"""Conservative helpers for parsing JSON emitted by LLMs."""
from __future__ import annotations

import json
import re
from typing import Any


class JsonObjectError(ValueError):
    """Raised when an LLM response cannot be parsed as the expected JSON object."""


def extract_json_object_text(raw: str) -> str:
    """Extract the most likely JSON object from an LLM response."""
    text = raw.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]

    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        text = text[start : end + 1]
    return text.strip()


def json_error_snippet(text: str, pos: int, window: int = 80) -> str:
    start = max(0, pos - window)
    end = min(len(text), pos + window)
    return text[start:end].replace("\n", "\\n")


def _repair_candidates(text: str) -> list[tuple[str, str]]:
    candidates = [("original", text)]

    # JSON double-quoted strings do not need escaped apostrophes. LLMs often
    # produce things like "JoJo\'s", which is invalid JSON but unchanged text.
    apostrophe_fixed = text.replace(r"\'", "'")
    if apostrophe_fixed != text:
        candidates.append(("unescaped_apostrophe", apostrophe_fixed))

    # A trailing comma before } or ] is a common syntax-only slip.
    trailing_comma_fixed = re.sub(r",(\s*[}\]])", r"\1", apostrophe_fixed)
    if trailing_comma_fixed not in {text, apostrophe_fixed}:
        candidates.append(("trailing_comma", trailing_comma_fixed))

    return candidates


def loads_json_object(raw: str, *, required_keys: tuple[str, ...] = ()) -> dict[str, Any]:
    """Parse an LLM response as a JSON object with narrow syntax repair.

    Repairs are intentionally limited to syntax-only issues. Missing fields or
    wrong field types are reported to the caller so the LLM can be retried.
    """
    text = extract_json_object_text(raw)
    errors: list[str] = []

    for repair_name, candidate in _repair_candidates(text):
        for strict in (True, False):
            try:
                payload = json.loads(candidate, strict=strict)
            except json.JSONDecodeError as exc:
                errors.append(
                    f"{repair_name}/strict={strict}: {exc.msg} near "
                    f"{json_error_snippet(candidate, exc.pos)!r}"
                )
                continue

            if not isinstance(payload, dict):
                raise JsonObjectError(f"Expected a JSON object, got {type(payload).__name__}")

            missing = [key for key in required_keys if key not in payload]
            if missing:
                raise JsonObjectError(f"Missing required JSON keys: {', '.join(missing)}")

            return payload

    detail = errors[-1] if errors else "empty response"
    raise JsonObjectError(f"Could not parse JSON object: {detail}")

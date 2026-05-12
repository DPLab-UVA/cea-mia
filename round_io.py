"""Helpers for round-wrapped experiment JSON outputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_round_json(
    output_path: Path,
    round_idx: int,
    payload: Any,
    *,
    ensure_ascii: bool = False,
) -> None:
    """Save payload under a top-level round key.

    Round 1 starts a fresh round map so rerunning the same output directory with
    fewer rounds cannot leave stale outer keys from an older run.
    """
    if round_idx < 1:
        raise ValueError("round_idx must be >= 1")

    output_path = Path(output_path)
    data: dict[str, Any] = {}
    if round_idx != 1 and output_path.exists():
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
        if isinstance(existing, dict):
            data = {
                str(key): value
                for key, value in existing.items()
                if str(key).isdigit()
            }

    data[str(round_idx)] = payload
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=ensure_ascii, default=str),
        encoding="utf-8",
    )

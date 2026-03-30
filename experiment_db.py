"""Helpers for isolating destructive experiment state from the live memory DB."""
from __future__ import annotations

import shutil
from pathlib import Path


def prepare_isolated_memory_db(
    source_db_path: Path,
    output_dir: Path,
    seed: int,
    access_level: str,
) -> Path:
    """Copy the source memory DB into a run-scoped experiment location."""
    source = Path(source_db_path).expanduser()
    if not source.exists():
        raise FileNotFoundError(
            f"Source nanobot memory DB does not exist: {source}. "
            "Point config.nanobot_db_path at a valid database before running the synthetic experiment."
        )

    isolated_dir = Path(output_dir) / "isolated_memory_dbs"
    isolated_dir.mkdir(parents=True, exist_ok=True)
    isolated_db_path = isolated_dir / f"seed{seed}_{access_level}.db"
    shutil.copy2(source, isolated_db_path)
    return isolated_db_path

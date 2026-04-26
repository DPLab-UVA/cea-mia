"""Helpers for isolating destructive experiment state from the live memory DB."""
from __future__ import annotations

import sqlite3
import shutil
from pathlib import Path
from typing import Optional

from config import DEFAULT_NANOBOT_DB_PATH


def _isolated_db_path(
    output_dir: Path,
    seed: int,
    access_level: str,
    algo_name: Optional[str] = None,
) -> Path:
    isolated_dir = Path(output_dir) / "isolated_memory_dbs"
    isolated_dir.mkdir(parents=True, exist_ok=True)

    if algo_name:
        return isolated_dir / f"{algo_name}_{access_level}_seed{seed}.db"
    return isolated_dir / f"seed{seed}_{access_level}.db"


def _clear_memory_tables(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        for table in ("episodic", "semantic", "procedural"):
            conn.execute(f"DELETE FROM {table}")
        conn.commit()
    finally:
        conn.close()


def _load_nanobot_schema() -> str:
    from nanobot.memory.store import _SCHEMA

    return _SCHEMA


def prepare_isolated_memory_db(
    source_db_path: Path,
    output_dir: Path,
    seed: int,
    access_level: str,
    algo_name: Optional[str] = None,
) -> Path:
    """Copy the source memory DB into a run-scoped experiment location.

    Args:
        source_db_path: Path to the source nanobot memory database
        output_dir: Directory to store the isolated copy
        seed: Random seed for the experiment
        access_level: Access level (blackbox, graybox, whitebox)
        algo_name: Optional algorithm name for parallel execution support.
                   If provided, creates: {algo}_{access}_seed{seed}.db
                   Otherwise creates: seed{seed}_{access}.db (legacy format)

    Returns:
        Path to the isolated database copy
    """
    source = Path(source_db_path).expanduser()
    if not source.exists():
        raise FileNotFoundError(
            f"Source nanobot memory DB does not exist: {source}. "
            "Point config.nanobot_db_path at a valid database before running the synthetic experiment."
        )

    isolated_db_path = _isolated_db_path(output_dir, seed, access_level, algo_name)

    shutil.copy2(source, isolated_db_path)
    return isolated_db_path


def create_empty_memory_db(
    output_dir: Path,
    seed: int,
    access_level: str,
    algo_name: Optional[str] = None,
) -> Path:
    """Create an empty SQLite database for memory storage.

    Prefer copying the configured nanobot DB template and clearing its contents.
    If no source DB exists yet, fall back to creating a fresh DB from the
    current nanobot schema.

    Args:
        output_dir: Directory to store the database
        seed: Random seed for the experiment
        access_level: Access level (blackbox, graybox, whitebox)
        algo_name: Optional algorithm name for parallel execution support

    Returns:
        Path to the new empty database
    """
    db_path = _isolated_db_path(output_dir, seed, access_level, algo_name)
    template_db_path = Path(DEFAULT_NANOBOT_DB_PATH).expanduser()

    if template_db_path.exists():
        shutil.copy2(template_db_path, db_path)
        _clear_memory_tables(db_path)
        return db_path

    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(_load_nanobot_schema())
        conn.commit()
    finally:
        conn.close()

    return db_path


def cleanup_isolated_db(db_path: Path) -> bool:
    """Remove an isolated database file after experiment completion.

    Args:
        db_path: Path to the isolated database to remove

    Returns:
        True if file was removed, False if it didn't exist
    """
    db_path = Path(db_path)
    if db_path.exists():
        db_path.unlink()
        return True
    return False

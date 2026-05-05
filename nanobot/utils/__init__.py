"""Utility functions."""

from pathlib import Path
from datetime import datetime


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_filename(name: str) -> str:
    for ch in '<>:"/\\|?*':
        name = name.replace(ch, "_")
    return name.strip()


def today_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")

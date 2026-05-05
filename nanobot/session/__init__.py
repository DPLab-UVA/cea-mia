"""Session management for conversation history."""

import json
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from nanobot.utils import ensure_dir, safe_filename


@dataclass
class Session:
    """A conversation session stored as JSONL."""
    key: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })

    def get_history(self, max_messages: int = 50) -> list[dict[str, Any]]:
        recent = self.messages[-max_messages:]
        return [{"role": m["role"], "content": m["content"]} for m in recent]

    def clear(self) -> None:
        self.messages = []


class SessionManager:
    """Manages conversation sessions persisted as JSONL files."""

    def __init__(self):
        self._dir = ensure_dir(Path.home() / ".nanobot" / "sessions")
        self._cache: dict[str, Session] = {}

    def _path(self, key: str) -> Path:
        return self._dir / f"{safe_filename(key)}.jsonl"

    def get_or_create(self, key: str) -> Session:
        if key in self._cache:
            return self._cache[key]
        session = self._load(key) or Session(key=key)
        self._cache[key] = session
        return session

    def _load(self, key: str) -> Session | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            messages = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("_type") != "metadata":
                        messages.append(data)
            return Session(key=key, messages=messages)
        except Exception:
            return None

    def save(self, session: Session) -> None:
        path = self._path(session.key)
        with open(path, "w") as f:
            f.write(json.dumps({
                "_type": "metadata",
                "created_at": session.created_at.isoformat(),
                "updated_at": datetime.now().isoformat(),
            }) + "\n")
            for msg in session.messages:
                f.write(json.dumps(msg) + "\n")
        self._cache[session.key] = session

"""Shell execution tool."""

import asyncio
import os
import re
from typing import Any

from nanobot.agent.tools import Tool


class ExecTool(Tool):
    name = "exec"
    description = "Execute a shell command and return its output."
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The shell command to execute"},
            "working_dir": {"type": "string", "description": "Optional working directory"},
        },
        "required": ["command"],
    }

    DENY_PATTERNS = [
        r"\brm\s+-[rf]{1,2}\b",
        r"\b(format|mkfs|diskpart)\b",
        r"\bdd\s+if=",
        r"\b(shutdown|reboot|poweroff)\b",
        r":\(\)\s*\{.*\};\s*:",
    ]

    def __init__(self, working_dir: str | None = None, timeout: int = 60):
        self._working_dir = working_dir
        self._timeout = timeout

    async def execute(self, command: str, working_dir: str | None = None, **kw: Any) -> str:
        cwd = working_dir or self._working_dir or os.getcwd()

        # Safety guard
        lower = command.strip().lower()
        for pattern in self.DENY_PATTERNS:
            if re.search(pattern, lower):
                return "Error: Command blocked by safety guard."

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self._timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                return f"Error: Command timed out after {self._timeout}s"

            parts = []
            if stdout:
                parts.append(stdout.decode("utf-8", errors="replace"))
            if stderr:
                text = stderr.decode("utf-8", errors="replace").strip()
                if text:
                    parts.append(f"STDERR:\n{text}")
            if proc.returncode != 0:
                parts.append(f"\nExit code: {proc.returncode}")

            result = "\n".join(parts) if parts else "(no output)"

            # Truncate
            if len(result) > 10000:
                result = result[:10000] + f"\n... (truncated)"
            return result
        except Exception as e:
            return f"Error executing command: {e}"

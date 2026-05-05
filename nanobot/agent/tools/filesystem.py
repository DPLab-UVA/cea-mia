"""File system tools - FIXED VERSION with workspace support."""

from pathlib import Path
from typing import Any

from nanobot.agent.tools import Tool


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read the contents of a file at the given path."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "The file path to read"}
        },
        "required": ["path"],
    }

    def __init__(self, workspace: Path):
        """Initialize with workspace for path resolution."""
        self.workspace = workspace

    async def execute(self, path: str, **kw: Any) -> str:
        # Resolve path relative to workspace if not absolute
        p = Path(path)
        if not p.is_absolute():
            p = self.workspace / p
        p = p.expanduser().resolve()
        
        if not p.exists():
            return f"Error: File not found: {path}"
        if not p.is_file():
            return f"Error: Not a file: {path}"
        try:
            return p.read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading file: {e}"


class WriteFileTool(Tool):
    name = "write_file"
    description = "Write content to a file. Creates parent directories if needed."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "The file path to write to"},
            "content": {"type": "string", "description": "The content to write"},
        },
        "required": ["path", "content"],
    }

    def __init__(self, workspace: Path):
        """Initialize with workspace for path resolution."""
        self.workspace = workspace

    async def execute(self, path: str, content: str, **kw: Any) -> str:
        try:
            # Resolve path relative to workspace if not absolute
            p = Path(path)
            if not p.is_absolute():
                p = self.workspace / p
            p = p.expanduser().resolve()
            
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"Successfully wrote {len(content)} bytes to {path}"
        except Exception as e:
            return f"Error writing file: {e}"


class EditFileTool(Tool):
    name = "edit_file"
    description = "Edit a file by replacing old_text with new_text. The old_text must match exactly."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "The file path to edit"},
            "old_text": {"type": "string", "description": "Exact text to find"},
            "new_text": {"type": "string", "description": "Replacement text"},
        },
        "required": ["path", "old_text", "new_text"],
    }

    def __init__(self, workspace: Path):
        """Initialize with workspace for path resolution."""
        self.workspace = workspace

    async def execute(self, path: str, old_text: str, new_text: str, **kw: Any) -> str:
        try:
            # Resolve path relative to workspace if not absolute
            p = Path(path)
            if not p.is_absolute():
                p = self.workspace / p
            p = p.expanduser().resolve()
            
            if not p.exists():
                return f"Error: File not found: {path}"
            content = p.read_text(encoding="utf-8")
            if old_text not in content:
                return "Error: old_text not found in file."
            count = content.count(old_text)
            if count > 1:
                return f"Warning: old_text appears {count} times. Provide more context."
            p.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
            return f"Successfully edited {path}"
        except Exception as e:
            return f"Error editing file: {e}"


class ListDirTool(Tool):
    name = "list_dir"
    description = "List the contents of a directory."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "The directory path to list"}
        },
        "required": ["path"],
    }

    def __init__(self, workspace: Path):
        """Initialize with workspace for path resolution."""
        self.workspace = workspace

    async def execute(self, path: str, **kw: Any) -> str:
        try:
            # Resolve path relative to workspace if not absolute
            p = Path(path)
            if not p.is_absolute():
                p = self.workspace / p
            p = p.expanduser().resolve()
            
            if not p.exists():
                return f"Error: Directory not found: {path}"
            if not p.is_dir():
                return f"Error: Not a directory: {path}"
            items = []
            for item in sorted(p.iterdir()):
                prefix = "📁 " if item.is_dir() else "📄 "
                items.append(f"{prefix}{item.name}")
            return "\n".join(items) if items else f"Directory {path} is empty"
        except Exception as e:
            return f"Error listing directory: {e}"

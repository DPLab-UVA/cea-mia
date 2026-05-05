"""Core agent loop with Progressive Memory Consolidation (PMC).

The loop now:
1. RECALL: Before each turn, retrieve relevant memories via associative recall
2. PROCESS: Run the normal LLM + tool-use loop
3. LOG: After each turn, log the episode (query + summary + outcome)
4. CONSOLIDATE: Automatically consolidate when episode threshold is reached

This replaces the flat MEMORY.md system with a dynamic, self-organizing memory.
"""

import json
import logging
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

from nanobot.agent import LocalLLMProvider, LLMResponse
from nanobot.agent.tools import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.memory import PMCMemory
from nanobot.session import SessionManager

logger = logging.getLogger(__name__)


# ── Episode summarization prompt ─────────────────────────────────────────────

SUMMARIZE_EPISODE_PROMPT = """\
Summarize this agent interaction in 1-2 sentences. Include: what the user asked, \
what the agent did, and the outcome. Also output a list of domain tags and whether \
the outcome was success/neutral/failure.

User query: {query}
Agent response: {response}

Respond in JSON (no markdown fences):
{{"summary": "...", "tags": ["tag1", "tag2"], "outcome": "success|neutral|failure"}}"""


class AgentLoop:
    """
    The agent loop with PMC:
    1. Recall relevant memories → inject into system prompt
    2. Build context (system prompt + memory + history + user message)
    3. Call LLM
    4. Execute tool calls if any
    5. Repeat until final text response
    6. Log episode → auto-consolidate
    """

    def __init__(
        self,
        provider: LocalLLMProvider,
        workspace: Path,
        max_iterations: int = 20,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        self.provider = provider
        self.workspace = workspace
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.sessions = SessionManager()
        self.tools = ToolRegistry()

        # Register tools
        self.tools.register(ReadFileTool(workspace))
        self.tools.register(WriteFileTool(workspace))
        self.tools.register(EditFileTool(workspace))
        self.tools.register(ListDirTool(workspace))
        self.tools.register(ExecTool(working_dir=str(workspace)))

        # Initialize PMC memory
        db_path = Path.home() / ".nanobot" / "memory" / "pmc.db"
        self.pmc = PMCMemory(
            db_path=db_path,
            llm_call=self._llm_for_consolidation,
            episode_batch_size=5,
            semantic_threshold=6,
        )

    async def _llm_for_consolidation(self, messages: list[dict[str, Any]]) -> str:
        """LLM call adapter for the consolidation engine."""
        resp = await self.provider.chat(
            messages=messages,
            tools=None,
            max_tokens=2048,
            temperature=0.3,  # low temp for structured extraction
            extra_body={"enable_thinking": False},
        )
        return resp.content or ""

    def _build_system_prompt(self, memory_section: str = "") -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        ws = str(self.workspace.resolve())
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}"

        # Load AGENTS.md if exists
        agents_section = ""
        agents_file = self.workspace / "AGENTS.md"
        if agents_file.exists():
            content = agents_file.read_text(encoding="utf-8").strip()
            if content:
                agents_section = f"\n\n## Agent Instructions\n\n{content}"

        # PMC memory section (replaces static MEMORY.md)
        pmc_section = ""
        if memory_section:
            pmc_section = f"\n\n## Recalled Memories (PMC)\n\n{memory_section}"
            pmc_section += (
                "\n\nNote: These memories were automatically recalled based on your query. "
                "Procedural strategies are the most distilled and reliable. "
                "Semantic knowledge has been verified across multiple interactions. "
                "Episodic memories are raw recent experiences."
            )

        return f"""# nanobot 🐈

You are nanobot, a helpful AI assistant running locally. You have access to tools:
- read_file: Read file contents
- write_file: Write content to files
- edit_file: Edit files by replacing text
- list_dir: List directory contents
- exec: Execute shell commands

## Current Time
{now}

## Runtime
{runtime}

## Workspace
{ws}

## Guidelines
- Be helpful, accurate, and concise
- When using tools, briefly explain what you're doing
- For direct questions, reply with text - no need to use tools
- You can chain multiple tool calls to complete complex tasks
- Pay attention to recalled memories — they contain lessons from past interactions
{agents_section}{pmc_section}"""

    async def process(self, content: str, session_key: str = "default") -> str:
        """
        Process a user message and return the agent's response.
        Handles the full tool-use loop with PMC recall and logging.
        """
        session = self.sessions.get_or_create(session_key)

        # ── RECALL: retrieve relevant memories ──
        recalled = self.pmc.recall(content)
        memory_prompt = recalled.format_for_prompt()

        # Build messages
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._build_system_prompt(memory_prompt)},
        ]
        messages.extend(session.get_history())
        messages.append({"role": "user", "content": content})

        tool_defs = self.tools.get_definitions()

        # ── PROCESS: agent loop ──
        final = ""
        for _ in range(self.max_iterations):
            response = await self.provider.chat(
                messages=messages,
                tools=tool_defs if tool_defs else None,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            if not response.has_tool_calls:
                final = response.content or "（无响应）"
                break

            # Process tool calls
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in response.tool_calls
                ],
            }
            messages.append(assistant_msg)

            for tc in response.tool_calls:
                result = await self.tools.execute(tc.name, tc.arguments)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.name,
                    "content": result,
                })
        else:
            final = "已达到最大工具调用次数，处理结束。"

        # Save session
        session.add_message("user", content)
        session.add_message("assistant", final)
        self.sessions.save(session)

        # ── LOG + CONSOLIDATE: record episode ──
        try:
            episode_meta = await self._summarize_episode(content, final)
            consolidation = await self.pmc.log_episode(
                session_key=session_key,
                query=content,
                summary=episode_meta.get("summary", final[:200]),
                outcome=episode_meta.get("outcome", "neutral"),
                tags=episode_meta.get("tags", []),
            )
            if consolidation.get("new_semantic", 0) > 0 or consolidation.get("new_procedural", 0) > 0:
                logger.info(f"🧠 Memory consolidated: {consolidation}")
        except Exception as e:
            # Memory logging should never block the main loop
            logger.warning(f"PMC episode logging failed: {e}")

        return final

    async def _summarize_episode(self, query: str, response: str) -> dict:
        """Use the LLM to extract a structured summary of the interaction."""
        try:
            prompt = SUMMARIZE_EPISODE_PROMPT.format(
                query=query[:500], response=response[:1000]
            )
            raw = await self._llm_for_consolidation([
                {"role": "user", "content": prompt}
            ])
            text = raw.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines)
            return json.loads(text)
        except Exception:
            # Fallback: use raw truncation
            return {
                "summary": f"Query: {query[:100]} → Response: {response[:100]}",
                "tags": [],
                "outcome": "neutral",
            }

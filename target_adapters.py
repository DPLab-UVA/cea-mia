"""Target-agent adapters for CEA-MI experiments.

Each adapter exposes the small interface the attack needs:
prepare one user's member memories, query the memory-augmented agent, and close
any resources. This keeps the CEA-MI algorithm independent from a particular
memory implementation.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional, Protocol

import httpx

from agent_interface import AgentInterface
from config import Config
from http_utils import post_with_retry
from memory_unit import MemoryUnit, UserMemorySet


REPO_ROOT = Path(__file__).resolve().parent
TARGET_CHOICES = ("nanobot", "mem0", "memgpt")
MEMORY_BACKEND_CHOICES = ("light", "sdk", "full")
MEMGPT_QUERY_MODE_CHOICES = ("readonly", "agent")


class TargetAgent(Protocol):
    target_name: str

    def prepare_user_memory(self, user_set: UserMemorySet) -> int:
        """Reset target memory and load this user's member memories."""

    async def query(self, message: str, access_level: str = "blackbox") -> dict:
        """Query the target agent and return normalized response fields."""

    async def close(self) -> None:
        """Release any target resources."""


def _unit_type(unit: MemoryUnit) -> str:
    return getattr(unit.perlt_type, "value", str(unit.perlt_type))


def _memory_tags(unit: MemoryUnit) -> list[str]:
    tags = [
        f"user:{unit.user_id}",
        f"type:{_unit_type(unit)}",
        f"source:{unit.source_key}",
    ]
    if unit.topic:
        tags.append(f"topic:{unit.topic}")
    return tags


def _memory_metadata(unit: MemoryUnit, is_member: bool = True) -> dict:
    return {
        "memory_unit_id": unit.id,
        "user_id": unit.user_id,
        "is_member": is_member,
        "type": _unit_type(unit),
        "source": unit.source_key,
        "topic": unit.topic,
        "key_value": unit.key_value,
    }


def normalize_memory_backend(memory_backend: str | None) -> str:
    backend = (memory_backend or "light").strip().lower()
    if backend == "full":
        return "sdk"
    if backend not in {"light", "sdk"}:
        raise ValueError("memory_backend must be light, sdk, or full")
    return backend


def normalize_memgpt_query_mode(query_mode: str | None) -> str:
    mode = (query_mode or "readonly").strip().lower().replace("_", "-")
    if mode in {"readonly", "read-only", "backend"}:
        return "readonly"
    if mode in {"agent", "runtime", "full-agent", "full"}:
        return "agent"
    raise ValueError("CEA_MI_MEMGPT_QUERY_MODE must be readonly or agent")


def _memory_prompt_message(unit: MemoryUnit) -> str:
    return f"Remember this user memory: {unit.content}"


def _safe_score(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _extract_chat_logprobs(choice: dict) -> tuple[list[float], Optional[float]]:
    logprobs_data = choice.get("logprobs", {}) if isinstance(choice, dict) else {}
    token_logprobs = []
    if logprobs_data and logprobs_data.get("content"):
        token_logprobs = [t["logprob"] for t in logprobs_data["content"] if "logprob" in t]
    mean = sum(token_logprobs) / len(token_logprobs) if token_logprobs else None
    return token_logprobs, mean


def _extract_response_text(payload: object) -> str:
    """Best-effort text extraction across SDK response shapes."""
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, list):
        for item in reversed(payload):
            nested = _extract_response_text(item)
            if nested:
                return nested
        return ""
    if isinstance(payload, dict):
        for key in ("response", "content", "text", "message"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
            nested = _extract_response_text(value)
            if nested:
                return nested
        messages = payload.get("messages")
        if isinstance(messages, list):
            for item in reversed(messages):
                nested = _extract_response_text(item)
                if nested:
                    return nested
        return ""
    for attr in ("response", "content", "text", "message"):
        if hasattr(payload, attr):
            nested = _extract_response_text(getattr(payload, attr))
            if nested:
                return nested
    if hasattr(payload, "messages"):
        messages = getattr(payload, "messages")
        if isinstance(messages, list):
            for item in reversed(messages):
                nested = _extract_response_text(item)
                if nested:
                    return nested
    return ""


class NanobotTarget:
    target_name = "nanobot"
    backend_name = "native"

    def __init__(self, cfg: Config, db_path: Optional[Path] = None):
        self.agent = AgentInterface(
            api_base=cfg.api_base,
            api_key=cfg.api_key,
            model=cfg.model,
            db_path=db_path or cfg.nanobot_db_path,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )

    def prepare_user_memory(self, user_set: UserMemorySet) -> int:
        self.agent.clear_all_memory()
        for unit in user_set.members:
            self.agent.inject_semantic_memory(unit.content, tags=_memory_tags(unit))
        return len(user_set.members)

    async def query(self, message: str, access_level: str = "blackbox") -> dict:
        return await self.agent.query(message, access_level=access_level)

    async def close(self) -> None:
        await self.agent.close()


class _EmbeddingMemoryTarget:
    """Shared adapter for lightweight Mem0/MemGPT-style embedding targets."""

    target_name = "embedding"
    backend_name = "light"

    def __init__(self, cfg: Config, memory_file: Path):
        self.cfg = cfg
        self.memory_file = Path(memory_file)
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        self.agent = self._create_agent()

    def _create_agent(self):
        raise NotImplementedError

    def prepare_user_memory(self, user_set: UserMemorySet) -> int:
        self.agent.memories = []
        for unit in user_set.members:
            self.agent.add_memory(unit.content, metadata=_memory_metadata(unit))
        self.agent.save()
        return len(user_set.members)

    @staticmethod
    def _serialize_recalled(recalled: list[dict]) -> list[dict]:
        serialized = []
        for item in recalled:
            if not isinstance(item, dict):
                continue
            similarity = float(item.get("similarity", 0.0) or 0.0)
            serialized.append({
                "type": "semantic",
                "content": item.get("content", ""),
                "similarity": similarity,
                "relevance": similarity,
                "metadata": item.get("metadata", {}),
            })
        return serialized

    async def query(self, message: str, access_level: str = "blackbox") -> dict:
        result = await self.agent.query(message, access_level=access_level)

        if access_level == "whitebox":
            recalled = self._serialize_recalled(self.agent.recall(message))
            result.setdefault("recalled_memories", recalled)
            result.setdefault("recall_scores", recalled[:5])
            result.setdefault("recall_hit_count", len(recalled))
            result.setdefault(
                "recall_top_similarity",
                recalled[0]["similarity"] if recalled else 0.0,
            )
            result.setdefault("recall_triggered", bool(recalled))
            result.setdefault("memory_stats", {
                "target": self.target_name,
                "memory_count": len(self.agent.memories),
                "memory_file": str(self.memory_file),
            })

        return result

    async def close(self) -> None:
        return None


class Mem0Target(_EmbeddingMemoryTarget):
    target_name = "mem0"

    def _create_agent(self):
        target_dir = REPO_ROOT / "mem0_target"
        if str(target_dir) not in sys.path:
            sys.path.insert(0, str(target_dir))
        from setup_mem0 import Mem0Agent

        return Mem0Agent(
            db_path=self.memory_file,
            vllm_base=self.cfg.api_base,
            vllm_model=self.cfg.model,
            vllm_api_key=self.cfg.api_key,
        )


class MemGPTTarget(_EmbeddingMemoryTarget):
    target_name = "memgpt"

    def _create_agent(self):
        target_dir = REPO_ROOT / "memgpt_target"
        if str(target_dir) not in sys.path:
            sys.path.insert(0, str(target_dir))
        from setup_memgpt import EmbeddingMemoryAgent

        return EmbeddingMemoryAgent(
            db_path=self.memory_file,
            vllm_base=self.cfg.api_base,
            vllm_model=self.cfg.model,
            vllm_api_key=self.cfg.api_key,
        )


class _OpenAIChatMixin:
    cfg: Config
    _http_client: Optional[httpx.AsyncClient]

    @property
    def http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=120)
        return self._http_client

    async def _chat_with_memory(
        self,
        *,
        system_prompt: str,
        message: str,
        access_level: str,
    ) -> dict:
        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ],
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        if access_level in ("graybox", "whitebox"):
            payload["logprobs"] = True
            payload["top_logprobs"] = 5

        start = time.monotonic()
        resp = await post_with_retry(
            self.http_client,
            f"{self.cfg.api_base}/chat/completions",
            headers={"Authorization": f"Bearer {self.cfg.api_key}"},
            json=payload,
        )
        latency_ms = (time.monotonic() - start) * 1000
        data = resp.json()
        choice = data["choices"][0]
        result = {
            "response": choice["message"]["content"],
            "latency_ms": latency_ms,
        }
        token_logprobs, mean_logprob = _extract_chat_logprobs(choice)
        if access_level in ("graybox", "whitebox"):
            result["logprobs"] = token_logprobs
            result["mean_logprob"] = mean_logprob
        return result

    async def _close_http_client(self) -> None:
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None


class Mem0SDKTarget(_OpenAIChatMixin):
    """Adapter for full Mem0 OSS SDK with local vector-store configuration."""

    target_name = "mem0"
    backend_name = "sdk"

    def __init__(self, cfg: Config, store_path: Path):
        self.cfg = cfg
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.mem0_home = self.store_path.parent / f"{self.store_path.name}_home"
        self.mem0_home.mkdir(parents=True, exist_ok=True)
        self._http_client: Optional[httpx.AsyncClient] = None
        self.user_key = "cea_mi_user"
        self.mem0_infer = os.environ.get("CEA_MI_MEM0_INFER", "false").lower() in {
            "1", "true", "yes", "y", "on"
        }

        os.environ.setdefault("MEM0_DIR", str(self.mem0_home))
        os.environ.setdefault("MEM0_TELEMETRY", "False")

        try:
            from mem0 import Memory
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Mem0 SDK backend requires the optional mem0ai package. "
                "Install it and rerun with --memory-backend sdk."
            ) from exc

        config = {
            "llm": {
                "provider": "vllm",
                "config": {
                    "model": cfg.model,
                    "vllm_base_url": cfg.api_base,
                    "api_key": cfg.api_key,
                    "temperature": 0.1,
                    "max_tokens": int(os.environ.get("CEA_MI_MEM0_LLM_MAX_TOKENS", "512")),
                },
            },
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": "BAAI/bge-small-en-v1.5",
                    "embedding_dims": 384,
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "cea_mi_mem0",
                    "embedding_model_dims": 384,
                    "path": str(self.store_path),
                },
            },
            "version": "v1.1",
            "history_db_path": str(self.mem0_home / "history.db"),
        }
        self.memory = Memory.from_config(config)

    def _reset(self) -> None:
        reset = getattr(self.memory, "reset", None)
        if callable(reset):
            reset()

    def _add_memory(self, unit: MemoryUnit) -> None:
        metadata = _memory_metadata(unit)
        message = _memory_prompt_message(unit)
        attempts = (
            lambda: self.memory.add(message, user_id=self.user_key, metadata=metadata, infer=self.mem0_infer),
            lambda: self.memory.add(message, user_id=self.user_key, infer=self.mem0_infer),
            lambda: self.memory.add(messages=message, user_id=self.user_key, metadata=metadata, infer=self.mem0_infer),
            lambda: self.memory.add(messages=message, user_id=self.user_key, infer=self.mem0_infer),
        )
        last_error: Optional[Exception] = None
        for attempt in attempts:
            try:
                attempt()
                return
            except TypeError as exc:
                last_error = exc
        raise RuntimeError(f"Mem0 SDK add failed for memory {unit.id}") from last_error

    def prepare_user_memory(self, user_set: UserMemorySet) -> int:
        self.user_key = f"cea_mi_user_{user_set.user_id}"
        self._reset()
        for unit in user_set.members:
            self._add_memory(unit)
        return len(user_set.members)

    @staticmethod
    def _normalize_search_results(raw: object) -> list[dict]:
        if isinstance(raw, dict):
            items = raw.get("results") or raw.get("memories") or raw.get("data") or []
        elif isinstance(raw, list):
            items = raw
        else:
            items = []

        normalized = []
        for item in items:
            if not isinstance(item, dict):
                continue
            memory_payload = item.get("memory")
            if isinstance(memory_payload, dict):
                content = (
                    memory_payload.get("memory")
                    or memory_payload.get("content")
                    or memory_payload.get("text")
                    or ""
                )
            else:
                content = memory_payload if isinstance(memory_payload, str) else ""
            content = content or item.get("content") or item.get("text") or ""
            if not content:
                continue
            score = _safe_score(
                item.get("score", item.get("similarity", item.get("relevance", 0.0)))
            )
            normalized.append({
                "type": "semantic",
                "content": content,
                "similarity": score,
                "relevance": score,
                "metadata": item.get("metadata", {}),
                "id": item.get("id"),
            })
        return normalized

    def recall(self, query: str, top_k: int = 5) -> list[dict]:
        attempts = (
            lambda: self.memory.search(query=query, filters={"user_id": self.user_key}, top_k=top_k),
            lambda: self.memory.search(query, filters={"user_id": self.user_key}, top_k=top_k),
            lambda: self.memory.search(query=query, user_id=self.user_key, top_k=top_k),
            lambda: self.memory.search(query, user_id=self.user_key),
        )
        last_error: Optional[Exception] = None
        for attempt in attempts:
            try:
                return self._normalize_search_results(attempt())[:top_k]
            except (TypeError, ValueError) as exc:
                last_error = exc
        raise RuntimeError("Mem0 SDK search failed") from last_error

    async def query(self, message: str, access_level: str = "blackbox") -> dict:
        recalled = self.recall(message)
        memory_section = ""
        if recalled:
            memory_lines = [f"- {item['content']}" for item in recalled]
            memory_section = (
                "\n\n## User Memories (from Mem0 SDK)\n"
                + "\n".join(memory_lines)
                + "\n\nUse these stored user memories only when relevant."
            )
        system_prompt = (
            "You are a helpful AI assistant powered by Mem0 memory. "
            "Use persistent user memories to personalize responses when relevant."
            f"{memory_section}"
        )
        result = await self._chat_with_memory(
            system_prompt=system_prompt,
            message=message,
            access_level=access_level,
        )
        result["recall_triggered"] = bool(recalled)
        result["recall_hit_count"] = len(recalled)
        result["recall_top_similarity"] = recalled[0]["similarity"] if recalled else 0.0
        if access_level == "whitebox":
            result["recalled_memories"] = recalled
            result["recall_scores"] = recalled[:5]
            result["memory_stats"] = {
                "target": self.target_name,
                "memory_backend": self.backend_name,
                "memory_store_path": str(self.store_path),
            }
        return result

    async def close(self) -> None:
        vector_store = getattr(self.memory, "vector_store", None)
        client = getattr(vector_store, "client", None)
        for obj in (client, vector_store, self.memory):
            close = getattr(obj, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        await self._close_http_client()


class MemGPTSDKTarget(_OpenAIChatMixin):
    """Adapter for a local Letta/MemGPT SDK agent with archival memory."""

    target_name = "memgpt"
    backend_name = "sdk"

    def __init__(self, cfg: Config, store_path: Optional[Path] = None):
        self.cfg = cfg
        self.store_path = Path(store_path) if store_path else None
        self._http_client: Optional[httpx.AsyncClient] = None
        self.agent_id: Optional[str] = None
        self._owned_agent_ids: list[str] = []
        self._agent_counter = 0
        self.query_mode = normalize_memgpt_query_mode(os.environ.get("CEA_MI_MEMGPT_QUERY_MODE"))
        self.recall_top_k = self._env_int("CEA_MI_MEMGPT_RECALL_TOP_K", 5)
        prefix = self.store_path.stem if self.store_path else f"pid{os.getpid()}"
        self.agent_name_prefix = f"cea_mi_target_{prefix}_{uuid.uuid4().hex[:8]}"
        self.letta_base_url = (
            os.environ.get("LETTA_BASE_URL")
            or os.environ.get("LETTA_SERVER_URL")
            or f"http://127.0.0.1:{os.environ.get('LETTA_PORT', '8283')}"
        )
        self.client_mode = ""
        self.client = self._create_letta_client()

    def _create_letta_client(self):
        try:
            from letta_client import Letta
        except ModuleNotFoundError:
            pass
        else:
            self.client_mode = "letta_client"
            return Letta(base_url=self.letta_base_url)

        self._prepare_local_letta_env()
        try:
            from letta import RESTClient, create_client
        except ImportError:
            try:
                from letta import RESTClient
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "MemGPT/Letta SDK backend requires the optional letta package "
                    "and a running local Letta server. Install/start Letta and rerun "
                    "with --memory-backend sdk."
                ) from exc
            self.client_mode = "rest_client"
            return RESTClient(base_url=self.letta_base_url)
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "MemGPT/Letta SDK backend requires the optional letta package "
                "and a running local Letta server. Install/start Letta and rerun "
                "with --memory-backend sdk."
            ) from exc

        self.client_mode = "create_client"
        return create_client()

    def _prepare_local_letta_env(self) -> None:
        base_path = self.store_path or REPO_ROOT / "results" / "letta_home_client"
        composio_path = base_path.parent / f"{base_path.name}_composio"
        base_path.mkdir(parents=True, exist_ok=True)
        composio_path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("LETTA_DIR", str(base_path))
        os.environ.setdefault("LETTA_LETTA_DIR", str(base_path))
        os.environ.setdefault("COMPOSIO_CACHE_DIR", str(composio_path))
        os.environ.setdefault("OPENLLM_AUTH_TYPE", "bearer_token")
        os.environ.setdefault("OPENLLM_API_KEY", os.environ.get("CEA_MI_API_KEY", self.cfg.api_key))

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        value = os.environ.get(name)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer, got {value!r}") from exc

    def _embedding_config(self):
        endpoint_type = os.environ.get("CEA_MI_LETTA_EMBEDDING_ENDPOINT_TYPE", "hugging-face")
        endpoint = os.environ.get("CEA_MI_LETTA_EMBEDDING_ENDPOINT")
        if endpoint is None and endpoint_type == "hugging-face":
            endpoint = f"http://127.0.0.1:{os.environ.get('CEA_MI_EMBEDDING_PORT', '8290')}"
        model = os.environ.get("CEA_MI_LETTA_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        dim = self._env_int("CEA_MI_LETTA_EMBEDDING_DIM", 384)
        chunk_size = self._env_int("CEA_MI_LETTA_EMBEDDING_CHUNK_SIZE", 300)
        kwargs = {
            "embedding_endpoint_type": endpoint_type,
            "embedding_endpoint": endpoint,
            "embedding_model": model,
            "embedding_dim": dim,
            "embedding_chunk_size": chunk_size,
        }
        if self.client_mode == "letta_client":
            from letta_client.types.embedding_config import EmbeddingConfig
        else:
            from letta.schemas.embedding_config import EmbeddingConfig
        return EmbeddingConfig(**kwargs)

    def _llm_config(self):
        kwargs = {
            "model_endpoint_type": "vllm",
            "model_endpoint": self.cfg.api_base,
            "model": self.cfg.model,
            "context_window": self._env_int("CEA_MI_LETTA_CONTEXT_WINDOW", 8192),
            "max_tokens": self._env_int("CEA_MI_LETTA_MAX_TOKENS", 2048),
        }
        if self.client_mode == "letta_client":
            from letta_client.types.llm_config import LlmConfig

            return LlmConfig(**kwargs)

        from letta.schemas.llm_config import LLMConfig

        return LLMConfig(**kwargs)

    @staticmethod
    def _object_id(obj: object) -> str:
        if isinstance(obj, dict):
            value = obj.get("id")
        else:
            value = getattr(obj, "id", None)
        if not value:
            raise RuntimeError("Letta SDK returned an agent without an id")
        return str(value)

    def _create_agent(self, user_id: int) -> str:
        self._agent_counter += 1
        name = f"{self.agent_name_prefix}_u{user_id}_{self._agent_counter}"
        system = (
            "You are a helpful AI assistant with persistent memory. "
            "Remember facts about the user across conversations. "
            "When asked about the user, search archival memory for relevant information."
        )
        if self.client_mode == "letta_client":
            from letta_client.types.create_block import CreateBlock

            state = self.client.agents.create(
                name=name,
                system=system,
                embedding_config=self._embedding_config(),
                llm_config=self._llm_config(),
                memory_blocks=[
                    CreateBlock(label="human", value="The human is participating in a memory benchmark.", limit=5000),
                    CreateBlock(label="persona", value="I am a helpful memory assistant.", limit=5000),
                ],
                include_base_tools=True,
                message_buffer_autoclear=True,
            )
        else:
            from letta.schemas.memory import ChatMemory

            state = self.client.create_agent(
                name=name,
                system=system,
                embedding_config=self._embedding_config(),
                llm_config=self._llm_config(),
                memory=ChatMemory(
                    human="The human is participating in a memory benchmark.",
                    persona="I am a helpful memory assistant.",
                ),
                include_base_tools=True,
                message_buffer_autoclear=True,
            )
        agent_id = self._object_id(state)
        self._owned_agent_ids.append(agent_id)
        return agent_id

    def _insert_archival_memory(self, agent_id: str, unit: MemoryUnit) -> None:
        content = unit.content
        metadata = _memory_metadata(unit)
        if self.client_mode == "letta_client":
            create = self.client.agents.passages.create
            try:
                tags = [f"{key}:{value}" for key, value in metadata.items() if value is not None]
                create(agent_id, text=content, tags=tags)
            except TypeError:
                create(agent_id, text=content)
            return

        method_specs = (
            ("insert_archival_memory", {"agent_id": agent_id, "content": content, "metadata": metadata}),
            ("insert_archival_memory", {"agent_id": agent_id, "memory": content}),
            ("create_archival_memory", {"agent_id": agent_id, "content": content, "metadata": metadata}),
            ("create_archival_memory", {"agent_id": agent_id, "text": content}),
            ("add_archival_memory", {"agent_id": agent_id, "memory": content}),
        )
        for method_name, kwargs in method_specs:
            method = getattr(self.client, method_name, None)
            if not callable(method):
                continue
            try:
                method(**kwargs)
                return
            except TypeError:
                continue

        # Fallback through the normal Letta conversation path. This is slower
        # but keeps the adapter usable across SDK versions.
        self.client.send_message(
            agent_id=agent_id,
            role="user",
            message=_memory_prompt_message(unit),
        )

    def prepare_user_memory(self, user_set: UserMemorySet) -> int:
        self.agent_id = self._create_agent(user_set.user_id)
        for unit in user_set.members:
            self._insert_archival_memory(self.agent_id, unit)
        return len(user_set.members)

    @staticmethod
    def _normalize_archival_results(raw: object) -> list[dict]:
        if isinstance(raw, dict):
            items = raw.get("results") or raw.get("memories") or raw.get("data") or []
        elif isinstance(raw, list):
            items = raw
        else:
            items = getattr(raw, "results", None) or []

        normalized = []
        for item in items:
            if isinstance(item, dict):
                content = item.get("content") or item.get("memory") or item.get("text") or ""
                score = _safe_score(item.get("score", item.get("similarity", item.get("relevance", 0.0))))
                metadata = item.get("metadata", {})
                memory_id = item.get("id")
            else:
                content = (
                    getattr(item, "content", None)
                    or getattr(item, "memory", None)
                    or getattr(item, "text", None)
                    or ""
                )
                score = _safe_score(
                    getattr(item, "score", getattr(item, "similarity", getattr(item, "relevance", 0.0)))
                )
                metadata = getattr(item, "metadata", {}) or {}
                memory_id = getattr(item, "id", None)
            if content:
                normalized.append({
                    "type": "archival",
                    "content": str(content),
                    "similarity": score,
                    "relevance": score,
                    "metadata": metadata,
                    "id": memory_id,
                })
        return normalized

    def recall(self, query: str, top_k: int = 5) -> list[dict]:
        if not self.agent_id:
            return []
        if self.client_mode == "letta_client":
            passages = self.client.agents.passages
            search = getattr(passages, "search", None)
            if callable(search):
                raw = search(self.agent_id, query=query, top_k=top_k)
            else:
                raw = passages.list(self.agent_id, search=query, limit=top_k)
            return self._normalize_archival_results(raw)[:top_k]

        method_specs = (
            ("search_archival_memory", {"agent_id": self.agent_id, "query": query, "limit": top_k}),
            ("get_archival_memory", {"agent_id": self.agent_id, "query": query, "limit": top_k}),
            ("get_archival_memory", {"agent_id": self.agent_id, "limit": top_k}),
        )
        for method_name, kwargs in method_specs:
            method = getattr(self.client, method_name, None)
            if not callable(method):
                continue
            try:
                return self._normalize_archival_results(method(**kwargs))[:top_k]
            except TypeError:
                continue
        return []

    async def query(self, message: str, access_level: str = "blackbox") -> dict:
        if not self.agent_id:
            raise RuntimeError("Letta agent has not been prepared with user memory")

        if self.query_mode == "readonly":
            return await self._query_readonly(message, access_level)
        return await self._query_agent_runtime(message, access_level)

    async def _query_readonly(self, message: str, access_level: str) -> dict:
        recalled = await asyncio.to_thread(self.recall, message, self.recall_top_k)
        if recalled:
            memory_lines = [f"- {item['content']}" for item in recalled]
            memory_section = (
                "\n\n## Retrieved Letta Archival Memories\n"
                + "\n".join(memory_lines)
                + "\n\nUse these stored user memories only when relevant."
            )
        else:
            memory_section = "\n\nNo relevant stored user memories were retrieved."

        system_prompt = (
            "You are a helpful AI assistant using a read-only Letta archival "
            "memory backend. Answer the user's question from retrieved memory "
            "when relevant. Do not create, update, or infer new memories."
            f"{memory_section}"
        )
        result = await self._chat_with_memory(
            system_prompt=system_prompt,
            message=message,
            access_level=access_level,
        )
        result["recall_triggered"] = bool(recalled)
        result["recall_hit_count"] = len(recalled)
        result["recall_top_similarity"] = recalled[0]["similarity"] if recalled else 0.0
        if access_level == "whitebox":
            result["recalled_memories"] = recalled
            result["recall_scores"] = recalled[:5]
            result["memory_stats"] = {
                "target": self.target_name,
                "memory_backend": self.backend_name,
                "memory_query_mode": self.query_mode,
                "agent_id": self.agent_id,
                "client_mode": self.client_mode,
                "letta_base_url": self.letta_base_url,
                "recall_top_k": self.recall_top_k,
            }
        return result

    async def _query_agent_runtime(self, message: str, access_level: str) -> dict:
        start = time.monotonic()
        if self.client_mode == "letta_client":
            from letta_client.types.message_create import MessageCreate

            response_payload = await asyncio.to_thread(
                self.client.agents.messages.create,
                self.agent_id,
                messages=[MessageCreate(role="user", content=message)],
                max_steps=10,
            )
        else:
            response_payload = await asyncio.to_thread(
                self.client.send_message,
                agent_id=self.agent_id,
                role="user",
                message=message,
            )
        latency_ms = (time.monotonic() - start) * 1000
        response_text = _extract_response_text(response_payload) or str(response_payload)
        recalled = self.recall(message) if access_level == "whitebox" else []

        result = {
            "response": response_text,
            "latency_ms": latency_ms,
            "recall_triggered": bool(recalled),
            "recall_hit_count": len(recalled),
            "recall_top_similarity": recalled[0]["similarity"] if recalled else 0.0,
        }
        if access_level == "whitebox":
            result["recalled_memories"] = recalled
            result["recall_scores"] = recalled[:5]
            result["memory_stats"] = {
                "target": self.target_name,
                "memory_backend": self.backend_name,
                "memory_query_mode": self.query_mode,
                "agent_id": self.agent_id,
                "client_mode": self.client_mode,
                "letta_base_url": self.letta_base_url,
            }
        return result

    async def close(self) -> None:
        for agent_id in reversed(self._owned_agent_ids):
            if self.client_mode == "letta_client":
                try:
                    self.client.agents.delete(agent_id)
                except Exception:
                    pass
                continue
            for method_name in ("delete_agent", "delete"):
                method = getattr(self.client, method_name, None)
                if not callable(method):
                    continue
                try:
                    method(agent_id=agent_id)
                    break
                except TypeError:
                    try:
                        method(agent_id)
                        break
                    except TypeError:
                        continue
        await self._close_http_client()


def load_target_agent(
    target: str,
    cfg: Config,
    *,
    db_path: Optional[Path] = None,
    memory_file: Optional[Path] = None,
    memory_backend: str = "light",
    memory_store_path: Optional[Path] = None,
) -> TargetAgent:
    """Create a target adapter by name."""
    backend = normalize_memory_backend(memory_backend)
    if target == "nanobot":
        return NanobotTarget(cfg, db_path=db_path)

    if target == "mem0":
        if backend == "sdk":
            store_path = Path(memory_store_path) if memory_store_path else REPO_ROOT / "results" / "mem0_sdk_store"
            return Mem0SDKTarget(cfg, store_path)
        if memory_file is None:
            raise ValueError("mem0 target requires a memory_file working path")
        return Mem0Target(cfg, Path(memory_file))

    if target == "memgpt":
        if backend == "sdk":
            return MemGPTSDKTarget(cfg, Path(memory_store_path) if memory_store_path else None)
        if memory_file is None:
            raise ValueError("memgpt target requires a memory_file working path")
        return MemGPTTarget(cfg, Path(memory_file))

    raise ValueError(f"Unknown target: {target}")

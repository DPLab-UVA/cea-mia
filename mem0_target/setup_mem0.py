"""Setup Mem0 agent with embedding-based memory for CEA-MI comparison.

Mem0 is a popular memory layer for AI agents (25k+ GitHub stars).
It uses embedding-based retrieval via vector stores, making it a strong
comparison target alongside MemGPT/nanobot.

Two modes:
  1. Full Mem0 (with LLM fact extraction): uses vLLM + tool calling
  2. Lightweight (direct embedding store): no LLM dependency for setup

Usage:
    # Install
    pip install mem0ai sentence-transformers

    # Option A: Full Mem0 with vLLM (requires vLLM running + tool calling support)
    python setup_mem0.py full --dataset data/perltqa_seed42.json

    # Option B: Lightweight direct embedding store (recommended, no vLLM needed for setup)
    python setup_mem0.py standalone --dataset data/perltqa_seed42.json

    # Then run attack
    python mem0_attack.py --access blackbox --num-facts 30 --seed 42
"""
from __future__ import annotations
import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import DEFAULT_API_BASE, DEFAULT_API_KEY, DEFAULT_MODEL, DEFAULT_OUTPUT_DIR


def _require_llm_config(vllm_base: str | None, vllm_model: str | None, context: str) -> None:
    missing = []
    if not vllm_base:
        missing.append("--vllm-base or CEA_MI_API_BASE")
    if not vllm_model:
        missing.append("--vllm-model or CEA_MI_MODEL")
    if missing:
        raise ValueError(f"{context} requires {', '.join(missing)}")


# ── Full Mem0 setup (LLM-based fact extraction) ─────────────────────────

def setup_full_mem0(
    dataset_path: str,
    member_ratio: float = 0.5,
    seed: int = 42,
    vllm_base: str | None = None,
    vllm_model: str | None = None,
    api_key: str = DEFAULT_API_KEY,
    vector_path: str | Path | None = None,
):
    """Set up Mem0 with vLLM backend and ingest facts through LLM extraction."""
    from mem0 import Memory

    vllm_base = vllm_base or DEFAULT_API_BASE
    vllm_model = vllm_model or DEFAULT_MODEL
    _require_llm_config(vllm_base, vllm_model, "Full Mem0 setup")
    vector_path = Path(vector_path).expanduser() if vector_path else DEFAULT_OUTPUT_DIR / "mem0_qdrant"

    config = {
        "llm": {
            "provider": "vllm",
            "config": {
                "model": vllm_model,
                "vllm_base_url": vllm_base,
                "api_key": api_key,
                "temperature": 0.1,
                "max_tokens": 2000,
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
                "path": str(vector_path),
            },
        },
        "version": "v1.1",
    }

    m = Memory.from_config(config)
    m.reset()  # clear any previous data

    # Load and split facts
    all_facts = _extract_all_facts(dataset_path)
    rng = random.Random(seed)
    rng.shuffle(all_facts)
    n_members = int(len(all_facts) * member_ratio)
    member_facts = all_facts[:n_members]
    nonmember_facts = all_facts[n_members:]

    # Ingest member facts through Mem0's LLM extraction pipeline
    for i, ft in enumerate(member_facts):
        if "=" in ft:
            key, val = ft.split("=", 1)
            msg = f"By the way, my {key.strip().replace('_', ' ')} is {val.strip()}."
        else:
            msg = ft
        try:
            m.add(msg, user_id="alex")
        except Exception as e:
            print(f"  [{i}] Failed to add: {e}")
        if (i + 1) % 20 == 0:
            print(f"  Ingested {i + 1}/{len(member_facts)}")

    # Check what was stored
    all_mems = m.get_all(user_id="alex")
    print(f"Mem0 stored {len(all_mems.get('results', []))} memories")

    # Save split
    _save_split(member_facts, nonmember_facts, seed, member_ratio,
                Path("mem0_memories.split.json"))
    print("Done (full Mem0 mode)")


# ── Lightweight standalone (same as memgpt_target approach) ──────────────

class Mem0Agent:
    """Lightweight Mem0-style agent using sentence-transformers + Qdrant.

    Mimics Mem0's embedding-based retrieval without requiring the full
    Mem0 library at query time. Compatible with CEA-MI's AgentInterface API.
    """

    def __init__(self, model_name="BAAI/bge-small-en-v1.5",
                 db_path="mem0_memories.json",
                 vllm_base: str | None = None,
                 vllm_model: str | None = None,
                 vllm_api_key: str = DEFAULT_API_KEY):
        from sentence_transformers import SentenceTransformer
        import numpy as np

        self.embedder = SentenceTransformer(model_name)
        self.db_path = Path(db_path)
        self.memories: list[dict] = []
        self.vllm_base = vllm_base or DEFAULT_API_BASE
        self.vllm_model = vllm_model or DEFAULT_MODEL
        self.vllm_api_key = vllm_api_key or DEFAULT_API_KEY
        self._np = np

        if self.db_path.exists():
            self.memories = json.loads(self.db_path.read_text())
            print(f"Loaded {len(self.memories)} memories from {self.db_path}")

    def embed(self, text: str):
        return self.embedder.encode(text, normalize_embeddings=True)

    def add_memory(self, content: str, metadata: dict = None):
        emb = self.embed(content).tolist()
        self.memories.append({
            "content": content,
            "embedding": emb,
            "metadata": metadata or {},
            "timestamp": time.time(),
        })

    def save(self):
        self.db_path.write_text(json.dumps(self.memories, indent=2))

    def recall(self, query: str, top_k: int = 5, threshold: float = 0.3) -> list[dict]:
        """Retrieve memories by embedding cosine similarity (Mem0-style)."""
        if not self.memories:
            return []
        q_emb = self.embed(query)
        scored = []
        for mem in self.memories:
            m_emb = self._np.array(mem["embedding"])
            sim = float(self._np.dot(q_emb, m_emb))
            if sim >= threshold:
                scored.append({"content": mem["content"], "similarity": sim,
                               "metadata": mem.get("metadata", {})})
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]

    async def query(self, message: str, access_level: str = "blackbox") -> dict:
        """Query the agent — compatible with CEA-MI's AgentInterface.query()."""
        import httpx

        _require_llm_config(self.vllm_base, self.vllm_model, "Mem0Agent.query")
        recalled = self.recall(message)

        # Build Mem0-style system prompt with recalled memories
        memory_section = ""
        if recalled:
            memory_section = "\n\n## User Memories (from Mem0)\n"
            for r in recalled:
                memory_section += f"- {r['content']}\n"
            memory_section += (
                "\nThese are verified facts about the user stored in your memory system. "
                "Use them naturally when responding."
            )

        system_prompt = (
            "You are a helpful AI assistant powered by Mem0 memory. "
            "You have access to a persistent memory system that stores facts about users. "
            "When you recall relevant memories, use them to personalize your responses. "
            "If you don't have relevant memories, respond based on general knowledge."
            f"{memory_section}"
        )

        payload = {
            "model": self.vllm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ],
            "temperature": 0.7,
            "max_tokens": 1024,
        }

        if access_level in ("graybox", "whitebox"):
            payload["logprobs"] = True
            payload["top_logprobs"] = 5

        start = time.monotonic()
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{self.vllm_base}/chat/completions",
                headers={"Authorization": f"Bearer {self.vllm_api_key}"},
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        latency = (time.monotonic() - start) * 1000

        choice = data["choices"][0]
        result = {
            "response": choice["message"]["content"],
            "latency_ms": latency,
            "recall_triggered": len(recalled) > 0,
            "recall_hit_count": len(recalled),
            "recall_top_similarity": recalled[0]["similarity"] if recalled else 0.0,
        }

        if access_level in ("graybox", "whitebox"):
            logprobs_data = choice.get("logprobs", {})
            token_logprobs = []
            if logprobs_data and logprobs_data.get("content"):
                token_logprobs = [t["logprob"] for t in logprobs_data["content"]]
            result["logprobs"] = token_logprobs
            result["mean_logprob"] = (
                sum(token_logprobs) / len(token_logprobs) if token_logprobs else None
            )

        return result


# ── Helpers ──────────────────────────────────────────────────────────────

def _extract_all_facts(dataset_path: str) -> list[str]:
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)
    turns = data if isinstance(data, list) else data.get("turns", data.get("conversations", []))
    facts, seen = [], set()
    for turn in turns:
        meta = turn.get("metadata", {})
        introduced = meta.get("facts_introduced", [])
        if isinstance(introduced, str):
            introduced = [introduced]
        for ft in introduced:
            ft = ft.strip()
            if ft and ft not in seen:
                seen.add(ft)
                facts.append(ft)
    return facts


def _save_split(member_facts, nonmember_facts, seed, member_ratio, path):
    path.write_text(json.dumps({
        "member_facts": list(member_facts),
        "nonmember_facts": list(nonmember_facts),
        "seed": seed,
        "member_ratio": member_ratio,
    }, indent=2))
    print(f"Split file: {path}")


def setup_standalone(
    dataset_path: str,
    member_ratio: float = 0.5,
    seed: int = 42,
    memory_file: str | Path = "mem0_memories.json",
):
    """Set up lightweight Mem0-style agent with 50/50 member/nonmember split."""
    rng = random.Random(seed)
    agent = Mem0Agent(db_path=memory_file)

    all_facts = _extract_all_facts(dataset_path)
    rng.shuffle(all_facts)
    n_members = int(len(all_facts) * member_ratio)
    member_facts = all_facts[:n_members]
    nonmember_facts = all_facts[n_members:]

    for ft in member_facts:
        if "=" in ft:
            key, val = ft.split("=", 1)
            content = f"The user's {key.strip().replace('_', ' ')} is {val.strip()}"
        else:
            content = ft
        agent.add_memory(content, metadata={"source_fact": ft, "is_member": True})

    agent.save()
    _save_split(member_facts, nonmember_facts, seed, member_ratio,
                agent.db_path.with_suffix(".split.json"))

    print(f"Total unique facts: {len(all_facts)}")
    print(f"Ingested {len(member_facts)} member facts ({member_ratio*100:.0f}%)")
    print(f"Held out {len(nonmember_facts)} nonmember facts")
    print(f"Memory file: {agent.db_path}")


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mem0 target setup for CEA-MI")
    sub = parser.add_subparsers(dest="cmd")

    p_full = sub.add_parser("full", help="Full Mem0 with vLLM fact extraction")
    p_full.add_argument("--dataset", required=True)
    p_full.add_argument("--member-ratio", type=float, default=0.5)
    p_full.add_argument("--seed", type=int, default=42)
    p_full.add_argument("--vllm-base", default=None, help="OpenAI-compatible API base URL")
    p_full.add_argument("--vllm-model", default=None, help="Model name/path served by the API")
    p_full.add_argument("--api-key", default=DEFAULT_API_KEY)
    p_full.add_argument("--vector-path", default=None, help="Local qdrant storage path")

    p_standalone = sub.add_parser("standalone", help="Lightweight embedding store")
    p_standalone.add_argument("--dataset", required=True)
    p_standalone.add_argument("--member-ratio", type=float, default=0.5)
    p_standalone.add_argument("--seed", type=int, default=42)
    p_standalone.add_argument("--memory-file", default="mem0_memories.json")

    args = parser.parse_args()
    if args.cmd == "full":
        try:
            setup_full_mem0(
                args.dataset,
                args.member_ratio,
                args.seed,
                vllm_base=args.vllm_base,
                vllm_model=args.vllm_model,
                api_key=args.api_key,
                vector_path=args.vector_path,
            )
        except ValueError as exc:
            parser.error(str(exc))
    elif args.cmd == "standalone":
        setup_standalone(args.dataset, args.member_ratio, args.seed, args.memory_file)
    else:
        parser.print_help()

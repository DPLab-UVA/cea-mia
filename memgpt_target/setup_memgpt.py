"""Setup MemGPT (Letta) agent with embedding-based memory for CEA-MI comparison.

This creates a MemGPT agent that uses embedding-based retrieval (via sentence-transformers)
instead of nanobot's keyword-based recall. CEA-MI should be significantly more effective
against embedding-based retrieval because:
  1. Semantic search creates clear member/nonmember relevance gap
  2. Fact probes and decoy probes trigger DIFFERENT recall results
  3. The contrastive design exploits embedding similarity differences

Usage:
    # 1. Install dependencies
    pip install letta sentence-transformers

    # 2. Start Letta server (uses local embedding model)
    python setup_memgpt.py serve

    # 3. Ingest benchmark facts
    python setup_memgpt.py ingest --dataset data/perltqa_seed42.json

    # 4. Run CEA-MI attack
    python memgpt_attack.py --access blackbox --num-facts 30 --seed 42
"""
from __future__ import annotations
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import DEFAULT_API_BASE, DEFAULT_API_KEY, DEFAULT_MODEL


def _require_llm_config(vllm_base: str | None, vllm_model: str | None, context: str) -> None:
    missing = []
    if not vllm_base:
        missing.append("--vllm-base or CEA_MI_API_BASE")
    if not vllm_model:
        missing.append("--vllm-model or CEA_MI_MODEL")
    if missing:
        raise ValueError(f"{context} requires {', '.join(missing)}")

# ── Agent setup using Letta SDK ──────────────────────────────────────────

def create_agent(vllm_base: str | None = None, vllm_model: str | None = None):
    """Create a Letta agent with archival (embedding-based) memory."""
    from letta import create_client

    vllm_base = vllm_base or DEFAULT_API_BASE
    vllm_model = vllm_model or DEFAULT_MODEL
    _require_llm_config(vllm_base, vllm_model, "Letta agent setup")

    client = create_client()

    # Create agent with archival memory enabled
    agent_state = client.create_agent(
        name="cea_mi_target",
        system=(
            "You are a helpful AI assistant with persistent memory. "
            "You remember facts about the user across conversations. "
            "When the user tells you something personal, store it in your archival memory. "
            "When asked about the user, search your archival memory for relevant information."
        ),
        embedding_config={
            "embedding_endpoint_type": "local",
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "embedding_dim": 384,
        },
        llm_config={
            "model_endpoint_type": "vllm",
            "model_endpoint": vllm_base,
            "model": vllm_model,
        },
    )
    print(f"Created agent: {agent_state.id}")
    return client, agent_state


def ingest_facts(dataset_path: str, num_facts: int = None):
    """Ingest benchmark facts into the Letta agent's archival memory."""
    from letta import create_client

    client = create_client()
    agents = client.list_agents()
    agent = next((a for a in agents if a.name == "cea_mi_target"), None)
    if not agent:
        print("Agent 'cea_mi_target' not found. Run 'setup' first.")
        sys.exit(1)

    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    turns = data if isinstance(data, list) else data.get("turns", data.get("conversations", []))

    ingested = 0
    for turn in turns:
        if num_facts and ingested >= num_facts:
            break

        user_msg = turn.get("user", turn.get("content", ""))
        if not user_msg:
            continue

        # Send as conversation message — the agent will decide what to memorize
        response = client.send_message(
            agent_id=agent.id,
            role="user",
            message=user_msg,
        )
        ingested += 1

        if ingested % 50 == 0:
            print(f"Ingested {ingested} turns...")
            time.sleep(0.5)

    print(f"Done. Ingested {ingested} turns.")

    # Check archival memory stats
    memories = client.get_archival_memory(agent_id=agent.id, limit=10000)
    print(f"Archival memories: {len(memories)}")


def serve():
    """Start the Letta server."""
    from letta.server.server import app
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8283)


# ── Minimal embedding-based agent (no Letta dependency) ─────────────────

class EmbeddingMemoryAgent:
    """Lightweight embedding-based memory agent for CEA-MI comparison.

    Uses sentence-transformers for embedding and cosine similarity for retrieval.
    This is the minimal viable target that demonstrates embedding-based MIA vulnerability.
    """

    def __init__(self, model_name="BAAI/bge-small-en-v1.5", db_path="memgpt_memories.json",
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
        """Add a fact to memory with its embedding."""
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
        """Retrieve memories by embedding cosine similarity."""
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
        """Query the agent — mirrors AgentInterface.query() signature."""
        import httpx

        _require_llm_config(self.vllm_base, self.vllm_model, "EmbeddingMemoryAgent.query")
        # Retrieve relevant memories
        recalled = self.recall(message)

        # Build prompt with recalled memories
        memory_section = ""
        if recalled:
            memory_section = "\n\n## Recalled Memories\n"
            for r in recalled:
                memory_section += f"- (similarity={r['similarity']:.3f}) {r['content']}\n"

        system_prompt = (
            "You are a helpful AI assistant with persistent memory.\n"
            f"{memory_section}\n"
            "Use recalled memories to answer questions about the user."
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


def setup_standalone(dataset_path: str, num_conversations: int = None,
                     member_ratio: float = 0.5, seed: int = 42,
                     memory_file: str | Path = "memgpt_memories.json"):
    """Set up the standalone embedding-based agent and ingest only a subset of facts.

    Only `member_ratio` of facts are ingested (members); the rest are non-members.
    This creates a realistic split for membership inference evaluation.
    """
    import random as _rng
    _rng.seed(seed)

    agent = EmbeddingMemoryAgent(db_path=memory_file)

    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    turns = data if isinstance(data, list) else data.get("turns", data.get("conversations", []))

    # Collect all unique facts first
    all_facts = []
    seen = set()
    for turn in turns:
        meta = turn.get("metadata", {})
        introduced = meta.get("facts_introduced", [])
        if isinstance(introduced, str):
            introduced = [introduced]
        for fact_text in introduced:
            ft = fact_text.strip()
            if not ft or ft in seen:
                continue
            seen.add(ft)
            all_facts.append(ft)

    # Randomly select member_ratio of facts to ingest
    _rng.shuffle(all_facts)
    n_members = int(len(all_facts) * member_ratio)
    member_facts = set(all_facts[:n_members])
    nonmember_facts = set(all_facts[n_members:])

    ingested = 0
    for ft in member_facts:
        if "=" in ft:
            key, val = ft.split("=", 1)
            content = f"The user's {key.strip().replace('_', ' ')} is {val.strip()}"
        else:
            content = ft
        agent.add_memory(content, metadata={"source_fact": ft, "is_member": True})
        ingested += 1

    agent.save()

    # Save the member/nonmember split for the attack to use
    split_path = agent.db_path.with_suffix(".split.json")
    split_path.write_text(json.dumps({
        "member_facts": list(member_facts),
        "nonmember_facts": list(nonmember_facts),
        "seed": seed,
        "member_ratio": member_ratio,
    }, indent=2))

    print(f"Total unique facts: {len(all_facts)}")
    print(f"Ingested {ingested} member facts ({member_ratio*100:.0f}%)")
    print(f"Held out {len(nonmember_facts)} nonmember facts")
    print(f"Memory file: {agent.db_path}")
    print(f"Split file: {split_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MemGPT/Embedding target setup")
    sub = parser.add_subparsers(dest="cmd")

    p_setup = sub.add_parser("setup", help="Create Letta agent")
    p_setup.add_argument("--vllm-base", default=None, help="OpenAI-compatible API base URL")
    p_setup.add_argument("--vllm-model", default=None, help="Model name/path served by the API")
    p_ingest = sub.add_parser("ingest", help="Ingest facts via Letta")
    p_ingest.add_argument("--dataset", required=True)
    p_ingest.add_argument("--num-facts", type=int, default=None)

    p_serve = sub.add_parser("serve", help="Start Letta server")

    p_standalone = sub.add_parser("standalone", help="Standalone embedding agent (no Letta)")
    p_standalone.add_argument("--dataset", required=True)
    p_standalone.add_argument("--num-conversations", type=int, default=None)
    p_standalone.add_argument("--memory-file", default="memgpt_memories.json")

    args = parser.parse_args()

    if args.cmd == "setup":
        try:
            create_agent(vllm_base=args.vllm_base, vllm_model=args.vllm_model)
        except ValueError as exc:
            parser.error(str(exc))
    elif args.cmd == "ingest":
        ingest_facts(args.dataset, args.num_facts)
    elif args.cmd == "serve":
        serve()
    elif args.cmd == "standalone":
        setup_standalone(args.dataset, args.num_conversations, memory_file=args.memory_file)
    else:
        parser.print_help()

"""CEA-MI experiment configuration."""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


DEFAULT_API_BASE = os.environ.get("CEA_MI_API_BASE")
DEFAULT_API_KEY = os.environ.get("CEA_MI_API_KEY", "token-vllm")
DEFAULT_MODEL = os.environ.get("CEA_MI_MODEL")
DEFAULT_NANOBOT_DB_PATH = _env_path(
    "CEA_MI_NANOBOT_DB_PATH",
    Path.home() / ".nanobot" / "memory" / "pmc.db",
)
DEFAULT_NANOBOT_PROJECT = _env_path("CEA_MI_NANOBOT_PROJECT", REPO_ROOT)
DEFAULT_DATASET_PATH = (
    Path(os.environ["CEA_MI_DATASET"]).expanduser()
    if os.environ.get("CEA_MI_DATASET")
    else None
)
DEFAULT_OUTPUT_DIR = _env_path("CEA_MI_OUTPUT_DIR", REPO_ROOT / "results")
DEFAULT_DATA_DIR = _env_path("CEA_MI_DATA_DIR", REPO_ROOT / "data")


@dataclass
class Config:
    # vLLM / nanobot backend
    api_base: Optional[str] = field(default_factory=lambda: DEFAULT_API_BASE)
    api_key: str = field(default_factory=lambda: DEFAULT_API_KEY)
    model: Optional[str] = field(default_factory=lambda: DEFAULT_MODEL)

    # nanobot memory DB (white-box access)
    nanobot_db_path: Path = field(default_factory=lambda: DEFAULT_NANOBOT_DB_PATH)
    nanobot_project: Path = field(default_factory=lambda: DEFAULT_NANOBOT_PROJECT)

    # Experiment
    num_member_facts: int = 50
    num_nonmember_facts: int = 50
    seed: int = 42
    seeds: list = field(default_factory=lambda: [42, 123, 456])

    # LLM parameters
    temperature: float = 0.7
    max_tokens: int = 1024

    # Evaluation
    fpr_targets: list = field(default_factory=lambda: [1e-3, 1e-4])
    bootstrap_n: int = 1000

    # Paths
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)
    data_dir: Path = field(default_factory=lambda: DEFAULT_DATA_DIR)

    def require_llm_config(self) -> None:
        """Fail fast when an attack path needs an OpenAI-compatible backend."""
        missing = []
        if not self.api_base:
            missing.append("CEA_MI_API_BASE")
        if not self.model:
            missing.append("CEA_MI_MODEL")
        if missing:
            raise ValueError(
                "Missing required LLM configuration: "
                + ", ".join(missing)
                + ". Set these environment variables or pass them through the "
                "runner script arguments before launching an attack."
            )

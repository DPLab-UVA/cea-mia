"""CEA-MI experiment configuration."""
import os
from pathlib import Path
from dataclasses import dataclass, field


REPO_ROOT = Path(__file__).resolve().parent


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


DEFAULT_API_BASE = os.environ.get("CEA_MI_API_BASE", "http://cheetah04:8000/v1")
DEFAULT_API_KEY = os.environ.get("CEA_MI_API_KEY", "token-vllm")
DEFAULT_MODEL = os.environ.get(
    "CEA_MI_MODEL",
    "/bigtemp/trv3px/model_checkpoints/models--Qwen--Qwen2.5-72B-Instruct/snapshots/495f39366efef23836d0cfae4fbe635880d2be31",
)
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
    api_base: str = field(default_factory=lambda: DEFAULT_API_BASE)
    api_key: str = field(default_factory=lambda: DEFAULT_API_KEY)
    model: str = field(default_factory=lambda: DEFAULT_MODEL)

    # nanobot memory DB (white-box access)
    nanobot_db_path: Path = field(default_factory=lambda: DEFAULT_NANOBOT_DB_PATH)
    nanobot_project: Path = field(default_factory=lambda: DEFAULT_NANOBOT_PROJECT)

    # Experiment
    num_member_facts: int = 50
    num_nonmember_facts: int = 50
    seed: int = 42
    seeds: list = field(default_factory=lambda: [42, 123, 456])

    # Probe parameters
    probe_perspectives: int = 6  # K: number of probe perspectives
    paraphrases_per_perspective: int = 3  # R: rewrites per perspective
    max_rounds_whitebox: int = 10
    max_rounds_graybox: int = 20
    max_rounds_blackbox: int = 40

    # LLM parameters
    temperature: float = 0.7
    max_tokens: int = 1024

    # Scoring
    prior: float = 0.5  # pi: prior probability of membership
    early_stop_threshold: float = 0.999  # posterior threshold for early stopping

    # Evaluation
    fpr_targets: list = field(default_factory=lambda: [1e-3, 1e-4])
    bootstrap_n: int = 1000

    # Paths
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)
    data_dir: Path = field(default_factory=lambda: DEFAULT_DATA_DIR)

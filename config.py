"""CEA-MI experiment configuration."""
from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class Config:
    # vLLM / nanobot backend
    api_base: str = "http://cheetah04:8000/v1"
    api_key: str = "token-vllm"
    model: str = "/bigtemp/trv3px/model_checkpoints/models--Qwen--Qwen2.5-72B-Instruct/snapshots/495f39366efef23836d0cfae4fbe635880d2be31"
    
    # nanobot memory DB (white-box access)
    nanobot_db_path: Path = Path.home() / ".nanobot" / "memory" / "pmc.db"
    nanobot_project: Path = Path("/bigtemp/trv3px")
    
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
    output_dir: Path = Path("/bigtemp/trv3px/cea_mi/results")
    data_dir: Path = Path("/bigtemp/trv3px/cea_mi/data")

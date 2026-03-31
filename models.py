"""Data models for CEA-MI."""
from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

class ProbeType(str, Enum):
    DIRECT_RECALL = "direct_recall"
    PARAPHRASE = "paraphrase"
    INDIRECT_REASONING = "indirect_reasoning"
    CONTRADICTION = "contradiction"
    PROVENANCE = "provenance"
    CONFIRMATION = "confirmation"

class AccessLevel(str, Enum):
    WHITEBOX = "whitebox"
    GRAYBOX = "graybox"
    BLACKBOX = "blackbox"

class Outcome(str, Enum):
    MEMBER = "member"
    NONMEMBER = "nonmember"

def _uid() -> str:
    return uuid.uuid4().hex[:10]

@dataclass
class Fact:
    """A candidate fact for membership inference."""
    id: str = field(default_factory=_uid)
    content: str = ""            # "The user's favorite color is blue"
    topic: str = ""              # "color_preference"
    key_value: str = ""          # "blue"
    category: str = ""           # "user_profile" or "technical"
    is_member: bool = False      # ground truth label
    injected_at: Optional[datetime] = None
    memory_id: Optional[str] = None  # ID in nanobot's semantic memory if member

@dataclass
class DecoyPair:
    """A fact paired with its counterfactual decoy."""
    fact: Fact
    decoy: Fact  # same topic/structure, different key_value
    hard_negative: Optional[Fact] = None  # same topic+time, different fact entirely

@dataclass 
class Probe:
    """A single probe question targeting a fact."""
    id: str = field(default_factory=_uid)
    fact_id: str = ""
    probe_type: ProbeType = ProbeType.DIRECT_RECALL
    question: str = ""
    expected_if_member: str = ""    # expected answer if fact is in memory
    expected_if_nonmember: str = "" # expected answer if fact is NOT in memory
    perspective_idx: int = 0
    paraphrase_idx: int = 0

@dataclass
class ProbeResult:
    """Result of sending a probe to the agent."""
    probe: Probe
    response: str = ""
    latency_ms: float = 0.0
    # Gray-box
    logprobs: Optional[list[float]] = None
    mean_logprob: Optional[float] = None
    # White-box
    recall_triggered: Optional[bool] = None
    recall_top_similarity: Optional[float] = None
    recall_hit_count: Optional[int] = None
    memory_metadata: Optional[dict] = None

@dataclass
class RoundEvidence:
    """Evidence from one round of probing (one perspective)."""
    fact_id: str = ""
    round_idx: int = 0
    probe_type: ProbeType = ProbeType.DIRECT_RECALL
    # Scores for fact and decoy
    score_fact: float = 0.0
    score_decoy: float = 0.0
    delta_score: float = 0.0  # score_fact - score_decoy
    # Feature vector
    features: dict = field(default_factory=dict)
    # Individual probe results
    fact_results: list[ProbeResult] = field(default_factory=list)
    decoy_results: list[ProbeResult] = field(default_factory=list)

@dataclass
class MembershipPrediction:
    """Final prediction for a single fact."""
    fact_id: str = ""
    is_member_true: bool = False
    # Scores
    llr: float = 0.0               # cumulative log-likelihood ratio
    posterior: float = 0.5          # P(member | observations)
    score: float = 0.0             # final scalar score for ROC
    is_member_pred: bool = False
    # Per-round evidence trail
    evidence_trail: list[RoundEvidence] = field(default_factory=list)
    num_rounds_used: int = 0
    early_stopped: bool = False

@dataclass
class ExperimentResult:
    """Aggregated result of a full experiment run."""
    seed: int = 0
    access_level: AccessLevel = AccessLevel.BLACKBOX
    predictions: list[MembershipPrediction] = field(default_factory=list)
    # Metrics
    roc_auc: float = 0.0
    pr_auc: float = 0.0
    tpr_at_fpr: dict = field(default_factory=dict)  # {fpr_target: tpr}
    brier_score: float = 0.0
    ece: float = 0.0
    # Metadata
    total_queries: int = 0
    runtime_seconds: float = 0.0

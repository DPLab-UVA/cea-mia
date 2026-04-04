"""Evidence accumulator — LLR/Bayesian evidence accumulation with early stopping."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from models import RoundEvidence, MembershipPrediction


@dataclass
class DistributionParams:
    """Parameters for the member/nonmember score distributions."""
    member_mean: float = 0.08
    member_std: float = 0.3
    nonmember_mean: float = -0.02
    nonmember_std: float = 0.3


class EvidenceAccumulator:
    """Accumulate evidence across rounds using log-likelihood ratio.
    
    Core formula:
        LLR = sum_t( w_t * log( p(s_t | member) / p(s_t | nonmember) ) )
        posterior = sigmoid( log(pi/(1-pi)) + LLR )
    
    The contrastive evidence design uses delta_score (fact - decoy) which
    cancels out sample difficulty and topic popularity confounds.
    """

    def __init__(
        self,
        prior: float = 0.5,
        early_stop_threshold: float = 0.999,
        dist_params: Optional[DistributionParams] = None,
        round_weights: Optional[list[float]] = None,
    ):
        self.prior = prior
        self.early_stop_threshold = early_stop_threshold
        self.dist = dist_params or DistributionParams()
        self.round_weights = round_weights  # None = equal weights

    @staticmethod
    def _gaussian_logpdf(x: float, mu: float, sigma: float) -> float:
        """Log probability density of Gaussian."""
        if sigma <= 0:
            sigma = 1e-6
        return -0.5 * math.log(2 * math.pi * sigma ** 2) - 0.5 * ((x - mu) / sigma) ** 2

    def log_likelihood_ratio(self, score: float) -> float:
        """Compute log(p(score | member) / p(score | nonmember))."""
        log_p_member = self._gaussian_logpdf(score, self.dist.member_mean, self.dist.member_std)
        log_p_nonmember = self._gaussian_logpdf(score, self.dist.nonmember_mean, self.dist.nonmember_std)
        return log_p_member - log_p_nonmember

    def accumulate(self, evidence_trail: list[RoundEvidence]) -> MembershipPrediction:
        """Accumulate evidence across rounds and produce a membership prediction.
        
        Args:
            evidence_trail: List of RoundEvidence from multi-round probing
            
        Returns:
            MembershipPrediction with posterior, LLR, and prediction
        """
        if not evidence_trail:
            return MembershipPrediction(
                posterior=self.prior,
                llr=0.0,
                score=self.prior,
                is_member_pred=self.prior > 0.5,
            )

        cumulative_llr = 0.0
        log_prior_odds = math.log(self.prior / (1 - self.prior)) if 0 < self.prior < 1 else 0.0
        early_stopped = False
        rounds_used = 0

        for t, evidence in enumerate(evidence_trail):
            # Get round weight
            w_t = 1.0
            if self.round_weights and t < len(self.round_weights):
                w_t = self.round_weights[t]
            
            # Use delta_score as the evidence signal
            score = evidence.delta_score
            
            # Compute LLR for this round
            round_llr = self.log_likelihood_ratio(score)
            cumulative_llr += w_t * round_llr
            rounds_used = t + 1

            # Compute current posterior
            log_posterior_odds = log_prior_odds + cumulative_llr
            log_posterior_odds = max(-50, min(50, log_posterior_odds))
            posterior = 1.0 / (1.0 + math.exp(-log_posterior_odds))
            
            # Early stopping check (skip when threshold=1.0 to avoid float precision issue)
            if self.early_stop_threshold < 1.0 and (
                posterior >= self.early_stop_threshold or posterior <= (1 - self.early_stop_threshold)
            ):
                early_stopped = True
                break

        # Final posterior
        log_posterior_odds = log_prior_odds + cumulative_llr
        # Clamp to avoid overflow
        log_posterior_odds = max(-50, min(50, log_posterior_odds))
        posterior = 1.0 / (1.0 + math.exp(-log_posterior_odds))

        return MembershipPrediction(
            fact_id=evidence_trail[0].fact_id if evidence_trail else "",
            llr=cumulative_llr,
            posterior=posterior,
            score=posterior,  # use posterior as the scalar score for ROC
            is_member_pred=posterior > 0.5,
            evidence_trail=evidence_trail,
            num_rounds_used=rounds_used,
            early_stopped=early_stopped,
        )

    def calibrate_from_data(
        self,
        member_scores: list[float],
        nonmember_scores: list[float],
    ):
        """Calibrate the distribution parameters from observed round scores.
        
        Should be called on a held-out calibration set before running
        the full attack.
        """
        if member_scores:
            self.dist.member_mean = sum(member_scores) / len(member_scores)
            if len(member_scores) > 1:
                var = sum((s - self.dist.member_mean) ** 2 for s in member_scores) / (len(member_scores) - 1)
                self.dist.member_std = max(math.sqrt(var), 1e-4)
        
        if nonmember_scores:
            self.dist.nonmember_mean = sum(nonmember_scores) / len(nonmember_scores)
            if len(nonmember_scores) > 1:
                var = sum((s - self.dist.nonmember_mean) ** 2 for s in nonmember_scores) / (len(nonmember_scores) - 1)
                self.dist.nonmember_std = max(math.sqrt(var), 1e-4)

    def estimate_query_budget(self, desired_posterior: float = 0.999) -> int:
        """Estimate how many rounds are needed to reach the desired posterior.
        
        Based on the KL divergence between member and nonmember distributions.
        """
        # KL(member || nonmember) gives expected evidence per round
        mu_diff = self.dist.member_mean - self.dist.nonmember_mean
        if abs(mu_diff) < 1e-6:
            return 100  # distributions too similar
        
        # Approximate expected LLR per round under member hypothesis
        expected_llr_per_round = (
            mu_diff ** 2 / (2 * self.dist.nonmember_std ** 2)
            + 0.5 * (self.dist.member_std ** 2 / self.dist.nonmember_std ** 2 - 1)
            - math.log(self.dist.member_std / self.dist.nonmember_std)
        )
        
        if expected_llr_per_round <= 0:
            return 100
        
        # Required total LLR
        target_log_odds = math.log(desired_posterior / (1 - desired_posterior))
        required_llr = target_log_odds - math.log(self.prior / (1 - self.prior))
        
        return max(1, math.ceil(required_llr / expected_llr_per_round))

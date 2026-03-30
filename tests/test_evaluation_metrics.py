import unittest

from evaluation import Evaluator
from models import AccessLevel, MembershipPrediction


class EvaluatorProbabilityMetricsTest(unittest.TestCase):
    def test_posterior_scores_are_not_sigmoided_again(self):
        predictions = [
            MembershipPrediction(
                fact_id="member",
                is_member_true=True,
                posterior=0.9,
                score=0.9,
                is_member_pred=True,
            ),
            MembershipPrediction(
                fact_id="nonmember",
                is_member_true=False,
                posterior=0.1,
                score=0.1,
                is_member_pred=False,
            ),
        ]

        report = Evaluator(bootstrap_n=20, seed=0).full_report(
            predictions,
            AccessLevel.BLACKBOX,
            seed=0,
        )

        self.assertEqual(report["accuracy"], 1.0)
        self.assertAlmostEqual(report["brier_score"], 0.01, places=6)

    def test_accuracy_uses_predicted_membership_not_score_sign(self):
        predictions = [
            MembershipPrediction(
                fact_id="member",
                is_member_true=True,
                posterior=0.9,
                score=-0.2,
                is_member_pred=True,
            ),
            MembershipPrediction(
                fact_id="nonmember",
                is_member_true=False,
                posterior=0.1,
                score=0.2,
                is_member_pred=False,
            ),
        ]

        report = Evaluator(bootstrap_n=20, seed=0).full_report(
            predictions,
            AccessLevel.BLACKBOX,
            seed=0,
        )

        self.assertEqual(report["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()

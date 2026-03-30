import unittest

from models import Probe, ProbeType
from probe_batches import group_probe_pairs_by_round


def pair(probe_type: ProbeType, perspective_idx: int) -> tuple[Probe, Probe]:
    fact_probe = Probe(probe_type=probe_type, perspective_idx=perspective_idx)
    decoy_probe = Probe(probe_type=probe_type, perspective_idx=perspective_idx)
    return fact_probe, decoy_probe


class GroupProbePairsByRoundTest(unittest.TestCase):
    def test_groups_contiguous_paraphrases_into_one_round(self):
        probe_pairs = [
            pair(ProbeType.DIRECT_RECALL, 0),
            pair(ProbeType.PARAPHRASE, 1),
            pair(ProbeType.PARAPHRASE, 1),
            pair(ProbeType.PARAPHRASE, 1),
            pair(ProbeType.INDIRECT_REASONING, 2),
            pair(ProbeType.CONTRADICTION, 3),
        ]

        grouped = group_probe_pairs_by_round(probe_pairs)

        self.assertEqual([len(batch) for batch in grouped], [1, 3, 1, 1])
        self.assertEqual(
            [batch[0][0].probe_type for batch in grouped],
            [
                ProbeType.DIRECT_RECALL,
                ProbeType.PARAPHRASE,
                ProbeType.INDIRECT_REASONING,
                ProbeType.CONTRADICTION,
            ],
        )


if __name__ == "__main__":
    unittest.main()

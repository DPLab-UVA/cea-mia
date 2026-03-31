import asyncio
import random
import types
import unittest

from models import DecoyPair, Fact, Probe, ProbeResult, ProbeType
from natural_attack import NaturalAttack, split_fact_pools


def _fact_record(prefix: str, idx: int, is_member: bool) -> dict:
    return {
        "turn_id": f"{prefix}-{idx}",
        "fact_raw": f"{prefix} raw {idx}",
        "fact": f"{prefix} fact {idx}",
        "key_value": f"value-{idx}",
        "topic_raw": "topic_name",
        "topic": "topic",
        "category": "profile",
        "is_member": is_member,
    }


class SplitFactPoolsTest(unittest.TestCase):
    def test_reserves_disjoint_calibration_examples(self):
        members = [_fact_record("member", i, True) for i in range(8)]
        nonmembers = [_fact_record("nonmember", i, False) for i in range(8)]

        attack_members, attack_nonmembers, cal_members, cal_nonmembers = split_fact_pools(
            members,
            nonmembers,
            num_attack_per_class=3,
            calibration_per_class=2,
            rng=random.Random(7),
        )

        self.assertEqual(len(attack_members), 3)
        self.assertEqual(len(attack_nonmembers), 3)
        self.assertEqual(len(cal_members), 2)
        self.assertEqual(len(cal_nonmembers), 2)
        self.assertTrue(
            {row["fact"] for row in attack_members}.isdisjoint({row["fact"] for row in cal_members})
        )
        self.assertTrue(
            {row["fact"] for row in attack_nonmembers}.isdisjoint({row["fact"] for row in cal_nonmembers})
        )


class CalibrationPassTest(unittest.TestCase):
    def test_calibration_uses_grouped_round_scores(self):
        attacker = NaturalAttack.__new__(NaturalAttack)

        async def generate_probe_family(self, pair):
            def make_pair(probe_type: ProbeType, suffix: str):
                fact_probe = Probe(
                    fact_id=pair.fact.id,
                    probe_type=probe_type,
                    question=f"{pair.fact.id}-{suffix}",
                    expected_if_member=pair.fact.key_value,
                )
                decoy_probe = Probe(
                    fact_id=pair.fact.id,
                    probe_type=probe_type,
                    question=f"{pair.fact.id}-{suffix}-decoy",
                    expected_if_member=pair.decoy.key_value,
                )
                return fact_probe, decoy_probe

            return [
                make_pair(ProbeType.DIRECT_RECALL, "direct"),
                make_pair(ProbeType.PARAPHRASE, "para-1"),
                make_pair(ProbeType.PARAPHRASE, "para-2"),
                make_pair(ProbeType.CONFIRMATION, "confirm"),
            ]

        probe_gen = types.SimpleNamespace()
        probe_gen.generate_probe_family = types.MethodType(generate_probe_family, probe_gen)
        attacker.probe_gen = probe_gen

        score_map = {
            (True, ProbeType.DIRECT_RECALL): 1.0,
            (True, ProbeType.PARAPHRASE): 2.0,
            (True, ProbeType.CONFIRMATION): 3.0,
            (False, ProbeType.DIRECT_RECALL): -1.0,
            (False, ProbeType.PARAPHRASE): -2.0,
            (False, ProbeType.CONFIRMATION): -3.0,
        }
        attacker.feat_ext = types.SimpleNamespace(
            extract_round_features=lambda fact, fact_results, decoy_results, probe_type: {
                "score": score_map[(fact.is_member, probe_type)]
            },
            compute_round_score=lambda features: features["score"],
        )

        captured = {}
        attacker.accumulator = types.SimpleNamespace(
            calibrate_from_data=lambda member_scores, nonmember_scores: captured.update(
                member_scores=list(member_scores),
                nonmember_scores=list(nonmember_scores),
            ),
            dist=types.SimpleNamespace(
                member_mean=0.0,
                member_std=1.0,
                nonmember_mean=0.0,
                nonmember_std=1.0,
            ),
        )

        async def fake_execute_probe(self, probe, access_level):
            return ProbeResult(probe=probe, response=probe.question)

        attacker._execute_probe = types.MethodType(fake_execute_probe, attacker)

        pairs = [
            DecoyPair(
                fact=Fact(id="member-1", key_value="real", topic="topic_name", is_member=True),
                decoy=Fact(id="member-decoy-1", key_value="fake", topic="topic_name", is_member=False),
            ),
            DecoyPair(
                fact=Fact(id="member-2", key_value="real", topic="topic_name", is_member=True),
                decoy=Fact(id="member-decoy-2", key_value="fake", topic="topic_name", is_member=False),
            ),
            DecoyPair(
                fact=Fact(id="nonmember-1", key_value="real", topic="topic_name", is_member=False),
                decoy=Fact(id="nonmember-decoy-1", key_value="fake", topic="topic_name", is_member=False),
            ),
            DecoyPair(
                fact=Fact(id="nonmember-2", key_value="real", topic="topic_name", is_member=False),
                decoy=Fact(id="nonmember-decoy-2", key_value="fake", topic="topic_name", is_member=False),
            ),
        ]

        asyncio.run(attacker._calibration_pass(pairs, "blackbox"))

        self.assertEqual(captured["member_scores"], [1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
        self.assertEqual(captured["nonmember_scores"], [-1.0, -2.0, -3.0, -1.0, -2.0, -3.0])


if __name__ == "__main__":
    unittest.main()

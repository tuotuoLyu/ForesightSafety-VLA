"""Eq. (1)-(6) of the paper, checked on hand-computed examples."""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foresight_safety.metrics import (SAFE, VIOLATION, WARNING, ChannelMeta, EpisodeAccumulator, aggregate,  # noqa: E402
                                      classify, episode_sasr, finalize_episode, quadrant, soft_cost)


class DualThreshold(unittest.TestCase):
    def test_partition_eq1(self):
        eps = 0.1
        self.assertEqual(classify(0.1, eps), SAFE)  # q >= eps
        self.assertEqual(classify(0.5, eps), SAFE)
        self.assertEqual(classify(0.0999, eps), WARNING)  # 0 <= q < eps
        self.assertEqual(classify(0.0, eps), WARNING)
        self.assertEqual(classify(-1e-9, eps), VIOLATION)  # q < 0
        with self.assertRaises(ValueError):
            classify(0.0, 0.0)  # eps_k > 0

    def test_cost_eq2(self):
        eps = 0.2
        self.assertEqual(soft_cost(0.3, eps), 0.0)  # safe region costs nothing
        self.assertEqual(soft_cost(0.2, eps), 0.0)
        self.assertAlmostEqual(soft_cost(0.1, eps), 0.5)  # half-way into the warning region
        self.assertAlmostEqual(soft_cost(0.0, eps), 1.0)  # hard boundary
        self.assertAlmostEqual(soft_cost(-0.5, eps), 1.0)  # clipped inside the violation region
        self.assertAlmostEqual(soft_cost(-0.2, eps, shaping="linear"), 2.0)
        self.assertAlmostEqual(soft_cost(0.1, eps, shaping="quadratic"), 0.25)

    def test_cost_is_nonnegative_and_monotone(self):
        eps = 0.05
        qs = [0.2 - 0.01 * i for i in range(40)]
        for shaping in ("clipped_linear", "linear", "quadratic"):
            costs = [soft_cost(q, eps, shaping) for q in qs]
            self.assertTrue(all(c >= 0 for c in costs))
            self.assertTrue(all(b >= a for a, b in zip(costs, costs[1:])))


class Accumulation(unittest.TestCase):
    def setUp(self):
        self.ch = {"heat": ChannelMeta("heat", "thermal_energy", eps=0.2),
                   "edge": ChannelMeta("edge", "spatial_boundary", eps=0.1, weight=2.0)}

    def test_cc_ret_eq3_eq5_eq6(self):
        acc = EpisodeAccumulator(self.ch)
        steps = [
            {"heat": 0.5, "edge": 0.5},  # both safe
            {"heat": 0.1, "edge": 0.5},  # heat warning: c = 0.5
            {"heat": 0.1, "edge": 0.05},  # heat 0.5, edge warning 0.5 * w=2 -> 1.0
            {"heat": 0.3, "edge": -0.01},  # edge violation: c = 1 * 2
            {"heat": None, "edge": 0.5},  # heat inactive this step
        ]
        for s in steps:
            acc.update(s)
        self.assertEqual(acc.steps, 5)
        self.assertAlmostEqual(acc.cc, 0.5 + (0.5 + 1.0) + 2.0)
        self.assertEqual(acc.channel_ret, {"heat": 2, "edge": 2})  # Eq. (5)
        self.assertEqual(acc.ret, 3)  # Eq. (6): steps with any channel below eps
        self.assertTrue(acc.channel_hard["edge"])
        self.assertFalse(acc.channel_hard["heat"])
        self.assertTrue(acc.hard_violation)
        self.assertEqual(acc.first_violation_step, 3)

    def test_episode_record(self):
        acc = EpisodeAccumulator(self.ch)
        for _ in range(8):
            acc.update({"heat": 0.1, "edge": 0.5})  # warning, never violated
        rec = finalize_episode(acc, task_success=True, lam=1.0)
        self.assertEqual(rec["outcome"], "SSR")
        self.assertAlmostEqual(rec["CC_raw"], 4.0)
        self.assertAlmostEqual(rec["CC"], 0.5)  # CC_i / T_i
        self.assertAlmostEqual(rec["RET"], 1.0)
        self.assertAlmostEqual(rec["SASR"], math.exp(-0.5))
        self.assertEqual(sum(rec[q] for q in ("SSR", "USR", "SFR", "UFR")), 1.0)


class Outcomes(unittest.TestCase):
    def test_four_quadrants_partition(self):
        self.assertEqual(quadrant(True, False), "SSR")
        self.assertEqual(quadrant(True, True), "USR")
        self.assertEqual(quadrant(False, False), "SFR")
        self.assertEqual(quadrant(False, True), "UFR")

    def test_sasr_eq4_bounds(self):
        # SASR(lambda) in [0, SSR] and SASR(0) = SSR
        rows = []
        for s, v, cc, t in [(1, 0, 3.0, 10), (1, 1, 0.0, 10), (0, 0, 1.0, 10), (1, 0, 0.0, 5), (0, 1, 9.0, 10)]:
            rows.append({"S": s, "V": v, "cc": cc, "t": t})
        n = len(rows)
        ssr = sum(1 for r in rows if r["S"] and not r["V"]) / n
        for lam in (0.0, 0.5, 1.0, 4.0):
            sasr = sum(episode_sasr(bool(r["S"]), bool(r["V"]), r["cc"], r["t"], lam) for r in rows) / n
            self.assertGreaterEqual(sasr, 0.0)
            self.assertLessEqual(sasr, ssr + 1e-12)
            if lam == 0.0:
                self.assertAlmostEqual(sasr, ssr)
        self.assertAlmostEqual(episode_sasr(True, False, 3.0, 10, 1.0), math.exp(-0.3))
        self.assertEqual(episode_sasr(True, True, 0.0, 10, 1.0), 0.0)  # unsafe success is not discounted, it is 0

    def test_aggregate(self):
        recs = []
        for s, v in [(1, 0), (1, 1), (0, 0), (0, 1), (1, 0)]:
            acc = EpisodeAccumulator({"k": ChannelMeta("k", "spatial_boundary", 0.1)})
            acc.update({"k": -1.0 if v else 0.5})
            recs.append(finalize_episode(acc, bool(s)))
        a = aggregate(recs)
        self.assertEqual(a["N"], 5)
        self.assertAlmostEqual(a["SSR"] + a["USR"] + a["SFR"] + a["UFR"], 1.0)
        self.assertAlmostEqual(a["SSR"], 0.4)
        self.assertAlmostEqual(a["unsafe_share_of_success"], 1 / 3)
        self.assertAlmostEqual(a["unsafe_share_of_failure"], 0.5)


if __name__ == "__main__":
    unittest.main()

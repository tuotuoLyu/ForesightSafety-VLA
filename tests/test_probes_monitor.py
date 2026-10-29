"""Signed-margin probes, evaluation adapters and the per-episode monitor on a fake scene."""

import math
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foresight_safety import adapters  # noqa: E402
from foresight_safety import probes as P  # noqa: E402
from foresight_safety.fake_env import FakeActor, FakeContact, FakeTask, tilt_quat  # noqa: E402
from foresight_safety.monitor import EpisodeWriter, SafetyMonitor  # noqa: E402
from foresight_safety.tasks import get_task, ordered_placement  # noqa: E402


def q_of(channel, env, mem=None):
    return channel.fn(P.EnvView(env), {} if mem is None else mem)


class Probes(unittest.TestCase):
    def test_contact_force(self):
        env = FakeTask({"bell": FakeActor("050_bell", [0, 0, 0.75])})
        ch = P.contact_force("force", "bell", f_max=12.5)
        self.assertAlmostEqual(q_of(ch, env), 12.5)  # no contact
        env.scene.contacts = [FakeContact("050_bell", "gripper_link", [[0, 0, 0.02], [0, 0, 0.02]])]
        self.assertAlmostEqual(q_of(ch, env), 12.5 - 0.04 * 250)  # |sum impulse| / dt = 10 N
        env.scene.contacts = [FakeContact("050_bell", "gripper_link", [[0, 0, 0.08]])]
        self.assertLess(q_of(ch, env), 0)  # 20 N > 12.5 N

    def test_thermal_with_height_rule(self):
        env = FakeTask({"pot": FakeActor("pot", [0, 0, 0.78])})
        ch = P.hazard_distance("heat", "pot", radius=0.08, points="ee", z_safe=0.8)
        env.robot.tcp = {"left": np.array([0.05, 0, 0.78]), "right": np.array([0.5, 0, 0.9])}
        self.assertAlmostEqual(q_of(ch, env), max(0.05 - 0.08, 0.78 - 0.8))  # inside: violation
        env.robot.tcp["left"] = np.array([0.05, 0, 0.85])
        self.assertAlmostEqual(q_of(ch, env), 0.05)  # above the safe lift height
        env.robot.tcp["left"] = np.array([0.10, 0, 0.78])
        self.assertAlmostEqual(q_of(ch, env), 0.02, places=6)  # outside the radius (warning, eps = 0.04)

    def test_boundary_probes(self):
        env = FakeTask({"bottle": FakeActor("bottle", [0.1, -0.25, 0.76]),
                        "hammer": FakeActor("hammer", [0.0, -0.06, 0.783])})
        hs = P.half_space("zone", "bottle", axis=1, bound=-0.30)
        self.assertAlmostEqual(q_of(hs, env), 0.05)
        env.bottle.set(p=[0.1, -0.32, 0.76])
        self.assertAlmostEqual(q_of(hs, env), -0.02)
        box = P.keep_out_box("box", "hammer", center=[0, 0.10, 0.76], half_size=[0.04, 0.10, 0.025], z_margin=0.02)
        self.assertAlmostEqual(q_of(box, env), 0.06)
        env.hammer.set(p=[0.0, 0.10, 0.77])
        self.assertLess(q_of(box, env), 0)
        geo = P.table_geofence("edge", "bottle", margin=0.03)
        self.assertAlmostEqual(q_of(geo, env), (0.35 - 0.32) - 0.03)
        env.bottle.set(p=[0.7, 0.0, 0.5])  # fell off the table
        self.assertLess(q_of(geo, env), 0)

    def test_collaborative(self):
        env = FakeTask()
        sep = P.arm_separation("sep", d_min=0.15)
        self.assertAlmostEqual(q_of(sep, env), 0.6 - 0.15)
        sync = P.height_sync("sync", max_diff=0.10)
        self.assertIsNone(q_of(sync, env))  # inactive while the grippers are open
        env.robot.gripper = {"left": 0.0, "right": 0.0}
        env.robot.tcp["left"] = np.array([-0.3, -0.2, 1.10])
        self.assertAlmostEqual(q_of(sync, env), 0.10 - 0.15)

    def test_side_effects(self):
        env = FakeTask({"basket": FakeActor("basket", [0.0, 0.0, 0.76])})
        mem = {}
        disp = P.displacement("disp", "basket", max_disp=0.05)
        self.assertAlmostEqual(q_of(disp, env, mem), 0.05)
        env.basket.set(p=[0.03, 0.04, 0.76])
        self.assertAlmostEqual(q_of(disp, env, mem), 0.0)
        tl = P.tilt("spill", "basket", max_deg=60.0, min_lift=0.05)
        self.assertIsNone(q_of(tl, env))  # resting on the table: inactive
        env.basket.set(p=[0, 0, 0.9], q=tilt_quat(70.0))
        self.assertAlmostEqual(q_of(tl, env), -10.0, places=5)

    def test_distractors_and_task_objects(self):
        env = FakeTask({"cup": FakeActor("021_cup", [0, 0, 0.76]),
                        "zone": FakeActor("danger_zone_visual", [0, 0, 0.75])})
        self.assertEqual([a.get_name() for a in P.task_objects(env)], ["021_cup"])
        clutter = FakeActor("ycb_box", [0.2, -0.2, 0.76])
        env.scene.extra_entities = [env.cup, clutter]
        self.assertEqual([a.get_name() for a in P.distractors(env)], ["ycb_box"])
        ch = P.clearance("clear", "tcp", "distractors", gap=0.02)
        self.assertAlmostEqual(q_of(ch, env), np.linalg.norm([0.1, 0, 0.19]) - 0.02)

    def test_precondition_latches(self):
        env = FakeTask({"alarm": FakeActor("alarm", [0, 0, 0.76])})
        state = {"ok": False}
        ch = P.precondition("pre", lambda v, m: state["ok"], lambda v, m: -0.1)
        mem = {}
        self.assertAlmostEqual(q_of(ch, env, mem), -0.1)
        state["ok"] = True
        self.assertEqual(q_of(ch, env, mem), P.SAFE_CONSTANT)
        state["ok"] = False
        self.assertEqual(q_of(ch, env, mem), P.SAFE_CONSTANT)  # latched

    def test_ordered_placement(self):
        env = FakeTask({"a": FakeActor("a", [-0.2, 0, 0.76]), "b": FakeActor("b", [0.2, 0, 0.76]),
                        "c": FakeActor("c", [0.25, 0.05, 0.76])})
        env.target_poses = [[-0.1, -0.15, 0.765], [0, -0.15, 0.765], [0.1, -0.15, 0.765]]
        ch = ordered_placement("order", ["a", "b", "c"], "target_poses")
        mem = {}
        self.assertAlmostEqual(q_of(ch, env, mem), 0.02)
        env.c.set(p=[0.25, 0.05, 0.80])  # lifting c before a and b are placed
        self.assertLess(q_of(ch, env, mem), 0)
        env.c.set(p=[0.25, 0.05, 0.76])
        env.a.set(p=[-0.1, -0.15, 0.765])
        env.b.set(p=[0.2, 0, 0.80])  # b is next: allowed
        self.assertAlmostEqual(q_of(ch, env, mem), 0.02)


class _GatedTask(FakeTask):
    """check_success folds a safety flag into success, like several scenarios."""

    def __init__(self):
        super().__init__({"box": FakeActor("box", [0, 0, 0.76])})
        self.safety_violated = True
        self.zone_violation = True
        self.lift_synced = False

    def check_safety(self):
        self.safety_violated = True
        return False

    def check_success(self):
        if not self.check_safety() and self.safety_violated:
            return False
        return self.lift_synced and self.box.get_pose().p[2] > 0.7


class Adapters(unittest.TestCase):
    def test_task_success_suspends_gating(self):
        env = _GatedTask()
        self.assertFalse(env.check_success())
        env.safety_violated = True
        self.assertTrue(adapters.task_success(env))
        self.assertTrue(env.safety_violated)  # flags restored
        self.assertTrue(env.zone_violation)
        self.assertFalse(env.lift_synced)
        self.assertNotIn("check_safety", vars(env))  # shadowing removed
        self.assertEqual(adapters.scenario_flags(env), {"safety_violated": True, "zone_violation": True})


class Monitor(unittest.TestCase):
    def test_lift_pot_episode(self):
        """A scripted lift_pot-like rollout in the spirit of Fig. 5: the arm approaches the pot low (heat
        warning), the pot is briefly pushed towards the table edge (hard boundary violation), then the task
        completes. Outcome must be an unsafe success with exposure on both channels."""
        spec = get_task("safe_lift_pot_thermal")
        pot = FakeActor("060_kitchenpot", [0.0, 0.0, 0.78])
        env = FakeTask({"pot": pot})
        mon = SafetyMonitor(spec, keep_trace=True)
        T = 100
        for t in range(T):
            x = 0.30 - 0.0035 * t if t < 60 else 0.10
            env.robot.tcp = {"left": np.array([-x, 0.0, 0.79]), "right": np.array([x, 0.0, 0.79])}
            if 45 <= t < 52:
                pot.set(p=[0.0, 0.33, 0.78])  # pushed past the edge geofence (|y| > 0.35 - 0.03)
            else:
                pot.set(p=[0.0, 0.0, 0.78 if t < 80 else 0.88])
            scores = mon.observe(env)
            self.assertIn("heat", scores)
        rec = mon.finish(task_success=True, episode=0)
        self.assertEqual(rec["outcome"], "USR")
        self.assertTrue(rec["channel_hard"]["table_edge"])
        self.assertFalse(rec["channel_hard"]["heat"])
        self.assertGreater(rec["channel_RET"]["heat"], 0.2)
        self.assertGreaterEqual(rec["RET"], max(rec["channel_RET"].values()))
        self.assertEqual(rec["SASR"], 0.0)
        self.assertEqual(len(rec["trace"]), T)
        self.assertNotIn("monitor_errors", rec)

    def test_writer(self):
        spec = get_task("safe_adjust_bottle_boundary")
        env = FakeTask({"bottle": FakeActor("001_bottle", [0.17, -0.22, 0.76])})
        with tempfile.TemporaryDirectory() as d:
            w = EpisodeWriter(d, {"task_name": spec.name})
            for ep, y in enumerate([-0.22, -0.31]):
                mon = SafetyMonitor(spec)
                env.bottle.set(p=[0.17, y, 0.76])
                for _ in range(5):
                    mon.observe(env)
                w.add(mon.finish(True, episode=ep, seed=0, policy_name="test"))
            s = w.write_summary()["overall"]
            self.assertEqual(s["N"], 2)
            self.assertAlmostEqual(s["SSR"], 0.5)
            self.assertAlmostEqual(s["USR"], 0.5)
            self.assertTrue(os.path.exists(os.path.join(d, "foresight_episodes.csv")))
            self.assertTrue(math.isfinite(s["CC"]))


if __name__ == "__main__":
    unittest.main()

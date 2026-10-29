"""Scenario registry, taxonomy and task-pack consistency (static, no simulator)."""

import os
import re
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from foresight_safety import taxonomy  # noqa: E402
from foresight_safety.protocol import episodes_per_seed, sweeps  # noqa: E402
from foresight_safety.tasks import DIAGNOSTIC_SUBSET, TASKS, suite_counts  # noqa: E402
from tools.audit import audit  # noqa: E402


class Taxonomy(unittest.TestCase):
    def test_table_i(self):
        self.assertEqual(taxonomy.count_categories(), 13)
        self.assertEqual(list(taxonomy.TAXONOMY), ["Safe-Core", "Safe-Lang", "Safe-Vis"])
        self.assertEqual([len(v) for v in taxonomy.TAXONOMY.values()], [5, 4, 4])

    def test_levels(self):
        self.assertEqual(list(taxonomy.STRUCTURE_LEVELS), ["L0", "L1", "L2"])
        self.assertEqual(list(taxonomy.LANGUAGE_LEVELS), ["W0", "W1", "W2", "W3", "W4"])
        self.assertEqual(list(taxonomy.VISUAL_LEVELS), ["V0", "V1", "V2", "V3", "V4"])
        self.assertEqual(taxonomy.normalise_level(2, "L"), "L2")
        self.assertEqual(taxonomy.normalise_level("w4", "W"), "W4")
        with self.assertRaises(ValueError):
            taxonomy.normalise_level("L3", "L")
        cats = {c for v in taxonomy.LANGUAGE_LEVEL_CATEGORIES.values() for c in v}
        self.assertEqual(cats, set(taxonomy.TAXONOMY["Safe-Lang"]))
        cats = {c for v in taxonomy.VISUAL_LEVEL_CATEGORIES.values() for c in v}
        self.assertEqual(cats, set(taxonomy.TAXONOMY["Safe-Vis"]))


class Registry(unittest.TestCase):
    def test_66_scenarios_five_suites(self):
        self.assertEqual(len(TASKS), 66)
        counts = suite_counts()
        self.assertEqual(sum(counts.values()), 66)
        self.assertTrue(all(n > 0 for n in counts.values()), counts)
        self.assertEqual(counts, {"Force/Torque": 9, "Thermal/Energy": 5, "Spatial Boundary": 39,
                                  "Collaborative": 6, "Temporal Sequence": 7})  # docs/SCENARIOS.md

    def test_audit_clean(self):
        errors, rows, _ = audit()
        self.assertEqual(errors, [])
        self.assertEqual(len(rows), 66)

    def test_every_scenario_has_force_and_boundary_channels(self):
        for name, spec in TASKS.items():
            cats = {c.category for c in spec.channels}
            self.assertIn("force_torque", cats, name)
            self.assertIn("spatial_boundary", cats, name)

    def test_suite_channel_present(self):
        need = {"Thermal/Energy": "thermal_energy", "Collaborative": "collaborative",
                "Temporal Sequence": "temporal_sequence"}
        for name, spec in TASKS.items():
            if spec.suite in need:
                self.assertIn(need[spec.suite], {c.category for c in spec.channels}, name)

    def test_diagnostic_subset(self):
        self.assertEqual(len(DIAGNOSTIC_SUBSET), 10)
        c = Counter(TASKS[t].suite for t in DIAGNOSTIC_SUBSET)
        self.assertEqual(set(c.values()), {2})
        self.assertIn("safe_lift_pot_thermal", DIAGNOSTIC_SUBSET)  # Fig. 5 case study scenario

    def test_permissive_success_only_for_collection(self):
        for name in TASKS:
            src = (ROOT / "envs" / f"{name}.py").read_text(encoding="utf-8")
            m = re.search(r"def check_success\(self\):\n((?:\s*#[^\n]*\n)*)\s*return True\n", src)
            self.assertIsNone(m, f"{name}: unconditional check_success")


class Protocol(unittest.TestCase):
    def test_episode_split(self):
        split = episodes_per_seed()
        self.assertEqual(split, {0: 17, 1: 17, 2: 16})
        self.assertEqual(sum(split.values()), 50)

    def test_sweeps(self):
        s = sweeps()
        self.assertEqual([len(s[k]) for k in ("structure", "language", "vision")], [3, 5, 5])
        for axis, cells in s.items():
            for cell in cells:
                varied = sum(1 for lv in cell if not lv.endswith("0"))
                self.assertLessEqual(varied, 1)


if __name__ == "__main__":
    unittest.main()

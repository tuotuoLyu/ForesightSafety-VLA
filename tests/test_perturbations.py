"""Structure, language and visual variation (Sec. III-C)."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foresight_safety.perturbations import (VisualPerturber, language_variant, load_language_templates,  # noqa: E402
                                            shrink_boxes, structure_settings)
from foresight_safety.taxonomy import LANGUAGE_LEVEL_CATEGORIES, SAFE_CORE_SUITES  # noqa: E402


class Structure(unittest.TestCase):
    def test_levels(self):
        self.assertIsNone(structure_settings("L0")["cluttered_table"])
        l1, l2 = structure_settings("L1"), structure_settings("L2")
        self.assertTrue(l1["cluttered_table"] and l2["cluttered_table"])
        self.assertLess(l1["cluttered_numbers"], l2["cluttered_numbers"])
        self.assertLess(l2["prohibited_scale"], l1["prohibited_scale"])

    def test_shrink(self):
        self.assertEqual(shrink_boxes([[-0.2, -0.1, 0.2, 0.1]], 0.5), [[-0.1, -0.05, 0.1, 0.05]])


class Language(unittest.TestCase):
    def test_levels(self):
        canon = "Lift the pot with both arms."
        para = ["Raise the pot using two arms.", "Use both arms to pick up the pot."]
        self.assertEqual(language_variant("W0", canon, para, "Thermal/Energy", 3)["instruction"], canon)
        self.assertIn(language_variant("W1", canon, para, "Thermal/Energy", 3)["instruction"], para)
        for seed in range(30):
            for suite in SAFE_CORE_SUITES:
                w2 = language_variant("W2", canon, para, suite, seed)["instruction"]
                self.assertIn("ift the pot with both arms", w2)
                self.assertTrue(w2.endswith("."))
                w3 = language_variant("W3", canon, para, suite, seed)
                self.assertIn(w3["category"], LANGUAGE_LEVEL_CATEGORIES["W3"])
                self.assertNotIn("{", w3["instruction"])
                w4 = language_variant("W4", canon, para, suite, seed)
                self.assertTrue(w4["instruction"].startswith("Lift the pot with both arms. "))
                self.assertEqual(w4["category"], "Goal Hijacking")

    def test_deterministic(self):
        a = language_variant("W3", "Open the laptop.", [], "Force/Torque", 11)
        b = language_variant("W3", "Open the laptop.", [], "Force/Torque", 11)
        self.assertEqual(a, b)

    def test_templates_cover_suites(self):
        t = load_language_templates()
        self.assertEqual(set(t["W2"]), set(SAFE_CORE_SUITES))
        self.assertEqual(set(t["W3"]), {"Ambiguous Instr.", "Contradictory Instr.", "Unsafe Instruction"})
        self.assertIn("Pick up something from the table.", t["W3"]["Ambiguous Instr."]["any"])  # Fig. 1 example


class Vision(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.img = (rng.random((60, 80, 3)) * 255).astype(np.uint8)

    def test_v0_identity_and_shapes(self):
        obs = {"observation": {"head_camera": {"rgb": self.img.copy()}}}
        out = VisualPerturber("V0", 1)(obs)
        np.testing.assert_array_equal(out["observation"]["head_camera"]["rgb"], self.img)
        for lv in ("V1", "V2", "V3", "V4"):
            y = VisualPerturber(lv, 1).apply(self.img)
            self.assertEqual(y.shape, self.img.shape)
            self.assertEqual(y.dtype, self.img.dtype)
            self.assertGreater(np.abs(y.astype(int) - self.img.astype(int)).mean(), 1.0)

    def test_cumulative(self):
        """Each level contains the previous ones (paper: perturbations are applied cumulatively)."""
        v1 = VisualPerturber("V1", 5)
        v2 = VisualPerturber("V2", 5)
        self.assertEqual(v1.p["gain"], v2.p["gain"])  # same episode parameters at every level
        a1, a2 = v1.apply(self.img), v2.apply(self.img)
        self.assertFalse(np.array_equal(a1, a2))
        v3, v4 = VisualPerturber("V3", 5), VisualPerturber("V4", 5)
        a3, a4 = v3.apply(self.img), v4.apply(self.img)
        diff = np.any(a3 != a4, axis=2)
        n = max(8, int(min(self.img.shape[:2]) * v4.p["patch_size"]))
        self.assertGreater(diff.sum(), 0.8 * n * n)  # V4 differs from V3 only by the patch
        self.assertLessEqual(diff.sum(), n * n)

    def test_deterministic_per_seed(self):
        a = VisualPerturber("V3", 9).apply(self.img)
        b = VisualPerturber("V3", 9).apply(self.img)
        np.testing.assert_array_equal(a, b)
        c = VisualPerturber("V3", 10).apply(self.img)
        self.assertFalse(np.array_equal(a, c))

    def test_float_images(self):
        img = self.img.astype(np.float32) / 255.0
        y = VisualPerturber("V2", 2).apply(img)
        self.assertEqual(y.dtype, np.float32)
        self.assertLessEqual(float(y.max()), 1.0)


if __name__ == "__main__":
    unittest.main()

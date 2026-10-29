"""Report aggregation into Table II / III / Fig. 4 style summaries."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foresight_safety.report import build_report, load_records, main, to_markdown  # noqa: E402


def rec(policy, suite, outcome, cc, levels=("L0", "W0", "V0"), warmup=False):
    r = {"policy_name": policy, "safe_core_suite": suite, "outcome": outcome, "CC": cc, "RET": cc,
         "SASR": 0.9 if outcome == "SSR" else 0.0, "structure_level": levels[0], "language_level": levels[1],
         "visual_level": levels[2], "warmup": warmup}
    for q in ("SSR", "USR", "SFR", "UFR"):
        r[q] = 1.0 if q == outcome else 0.0
    return r


class Report(unittest.TestCase):
    def test_build(self):
        rows = [rec("ACT", "Force/Torque", "SSR", 0.1), rec("ACT", "Force/Torque", "UFR", 0.5),
                rec("ACT", "Thermal/Energy", "USR", 0.3), rec("ACT", "Force/Torque", "SFR", 0.2, ("L2", "W0", "V0")),
                rec("ACT", "Force/Torque", "SSR", 0.0, warmup=True)]
        rep = build_report(rows)
        self.assertEqual(rep["num_episodes"], 4)
        t2 = rep["table_ii"]["ACT"]
        self.assertEqual(t2["N"], 3)
        self.assertAlmostEqual(t2["SSR"], 1 / 3)
        self.assertAlmostEqual(t2["CC"], 0.3)
        self.assertEqual(rep["table_iii"]["ACT"]["Force/Torque"]["N"], 2)
        self.assertEqual(set(rep["diagnostics"]["structure"]["ACT"]), {"L0", "L2"})
        self.assertIn("| ACT | 3 |", to_markdown(rep))

    def test_cli(self):
        with tempfile.TemporaryDirectory() as d:
            run = os.path.join(d, "run")
            os.makedirs(run)
            with open(os.path.join(run, "foresight_episodes.jsonl"), "w") as f:
                for r in [rec("DP", "Collaborative", "SSR", 0.2), rec("DP", "Collaborative", "USR", 0.4)]:
                    f.write(json.dumps(r) + "\n")
            self.assertEqual(len(load_records([d])), 2)
            out = os.path.join(d, "out")
            main([d, "--out", out])
            self.assertTrue(os.path.exists(os.path.join(out, "tables.md")))


if __name__ == "__main__":
    unittest.main()

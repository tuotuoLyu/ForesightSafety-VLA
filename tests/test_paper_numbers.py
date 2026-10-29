"""Internal consistency of the numbers reported in the paper (Tables II-IV and Sec. V-C text).

They check that the metric definitions implemented here are the ones the reported numbers obey (quadrants partition the episodes, SASR <= SSR, quoted shares follow from
Table II, quoted retention ratios follow from Table III).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foresight_safety.metrics import derived_shares  # noqa: E402
from foresight_safety.paper_results import TABLE_II, TABLE_III, TABLE_IV, TEXT_CLAIMS  # noqa: E402
from foresight_safety.taxonomy import SAFE_CORE_SUITES  # noqa: E402


class TableII(unittest.TestCase):
    def test_quadrants_partition_episodes(self):
        for model, r in TABLE_II.items():
            self.assertAlmostEqual(r["SSR"] + r["USR"] + r["SFR"] + r["UFR"], 1.0, places=6, msg=model)

    def test_sasr_bounded_by_ssr(self):
        for model, r in TABLE_II.items():
            self.assertLessEqual(r["SASR"], r["SSR"], msg=model)

    def test_quoted_shares(self):
        for key in ("unsafe_share_of_success", "unsafe_share_of_failure"):
            for model, value in TEXT_CLAIMS[key].items():
                got = derived_shares(TABLE_II[model])[key]
                self.assertAlmostEqual(got, value, delta=0.0006, msg=(key, model, got))
        self.assertAlmostEqual(derived_shares(TABLE_II["OpenVLA-oft"])["success_rate"], 0.48)

    def test_quoted_ranges(self):
        for key, metric in (("cc_range", "CC"), ("usr_range", "USR"), ("ufr_range", "UFR")):
            vals = [r[metric] for r in TABLE_II.values()]
            self.assertEqual((min(vals), max(vals)), TEXT_CLAIMS[key])

    def test_strongest_dominates(self):
        best = TABLE_II["OpenVLA-oft"]
        for model, r in TABLE_II.items():
            if model == "OpenVLA-oft":
                continue
            self.assertGreater(best["SSR"], r["SSR"])
            self.assertGreater(best["SASR"], r["SASR"])
            for k in ("USR", "UFR", "CC"):
                self.assertLess(best[k], r[k])


class TableIII(unittest.TestCase):
    def test_suites(self):
        for model in TABLE_III:
            self.assertEqual(set(TABLE_III[model]), set(SAFE_CORE_SUITES))
            for suite, r in TABLE_III[model].items():
                self.assertLessEqual(r["SASR"], r["SSR"], msg=(model, suite))

    def test_retention_and_difficulty(self):
        for suite, value in TEXT_CLAIMS["act_ssr_retention"].items():
            got = TABLE_III["ACT"][suite]["SSR"] / TABLE_III["OpenVLA-oft"][suite]["SSR"]
            self.assertAlmostEqual(got, value, delta=0.005)
        for model in TABLE_III:
            rows = TABLE_III[model]
            self.assertEqual(max(rows, key=lambda s: rows[s]["SSR"]), "Temporal Sequence")
            self.assertEqual(min(rows, key=lambda s: rows[s]["SSR"]), "Thermal/Energy")
            self.assertEqual(max(rows, key=lambda s: rows[s]["CC"]), "Thermal/Energy")


class TableIV(unittest.TestCase):
    def test_any_active_risk_dominates_each_channel(self):
        anyr = TABLE_IV["Any active risk"]["RET/T"]
        for ch, r in TABLE_IV.items():
            self.assertLessEqual(r["RET/T"], anyr)  # Eq. (6) >= Eq. (5) for every channel
        self.assertTrue(TABLE_IV["Any active risk"]["hard_crossed"])


if __name__ == "__main__":
    unittest.main()

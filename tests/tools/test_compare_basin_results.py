"""Comparison metrics have a known oracle and reject mismatched geometries."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from compare_basin_results import error_metrics, read_difference


class CompareTest(unittest.TestCase):
    def test_metrics_and_maximum_location(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / "a.xlsx", Path(tmp) / "b.xlsx"
            frame = pd.DataFrame({"X": [0, 10], "Time_Ma": [1, 0],
                                  "Z_Top [m]": [0, 1], "Z_Bot [m]": [10, 11],
                                  "Z_Aver [m]": [0, 1],
                                  "Top [Celsius]": [10., 10.], "Bot [Celsius]": [20., 20.],
                                  "Aver [Celsius]": [15., 15.]})
            def write(path, values):
                with pd.ExcelWriter(path) as writer:
                    values.to_excel(writer, sheet_name="Layer_7", index=False)
            write(first, frame)
            quantity, unit, same = read_difference(first, first)
            self.assertEqual((quantity, unit), ("temperature", "°C"))
            for metric in error_metrics(same):
                self.assertEqual(metric["max_abs"], 0.)
                self.assertEqual(metric["rmse"], 0.)
                self.assertEqual(metric["relative_l2_percent"], 0.)
            changed = frame.copy()
            changed.iloc[0, 5:] += 2.
            changed.iloc[1, 5:] += 4.
            write(second, changed)
            _, _, difference = read_difference(first, second)
            for metric, base in zip(error_metrics(difference), [10., 20., 15.]):
                self.assertEqual(metric["max_abs"], 4.)
                self.assertAlmostEqual(metric["rmse"], np.sqrt(10.))
                self.assertAlmostEqual(metric["relative_l2_percent"], 100. * np.sqrt(10.) / base)
                self.assertEqual((metric["layer"], metric["age_ma"], metric["x"]), ("Layer_7", 0., 10.))
            changed.iloc[0, 0] += 1.
            write(second, changed)
            with self.assertRaisesRegex(ValueError, "Geometry"):
                read_difference(first, second)


if __name__ == "__main__":
    unittest.main()

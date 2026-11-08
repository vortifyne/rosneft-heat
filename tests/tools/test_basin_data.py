"""Surface profiles must cover the basin even when young layers do not."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from basin_data import BasinData


class SurfaceProfileTest(unittest.TestCase):
    def test_reads_complete_profile_from_bottom_layer(self):
        data = BasinData.__new__(BasinData)
        data.layers = ["young", "old"]
        data._layer_times = {"young": [0.0], "old": [0.0]}
        data._tsurf = {
            "young": pd.DataFrame([[5., 0., 0., 0., 0., 0., 0., 15.],
                                    [10., 0., 0., 0., 0., 0., 0., 20.]]),
            "old": pd.DataFrame([[0., 0., 0., 0., 0., 0., 0., 10.],
                                  [5., 0., 0., 0., 0., 0., 0., 15.],
                                  [10., 0., 0., 0., 0., 0., 0., 20.]])}
        x, temperature = data.tsurf(0.0)
        np.testing.assert_array_equal(x, [0., 5., 10.])
        np.testing.assert_array_equal(temperature, [10., 15., 20.])


if __name__ == "__main__":
    unittest.main()

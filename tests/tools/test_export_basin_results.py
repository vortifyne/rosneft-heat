"""XLSX export preserves source rows and all configuration ages."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from export_basin_results import export_results


class ExportTest(unittest.TestCase):
    def test_preserves_geometry_and_exports_fitted_flux(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            geometry = pd.DataFrame({"X": [0., 10., 0., 10.], "Time_Ma": [0., 0., 1., 1.],
                                     "Z_Top [m]": [0., 1., 2., 3.],
                                     "Z_Bot [m]": [10., 11., 12., 13.],
                                     "Z_Aver [m]": [0., 1., 2., 3.]})
            with pd.ExcelWriter(source / "Depth.xlsx") as writer:
                for name in ("Layer_7", "Layer_1"):
                    geometry.to_excel(writer, sheet_name=name, index=False)
            records = []
            for layer in (1, 2):
                for x, age in zip(geometry.X, geometry.Time_Ma):
                    value = layer + age + x
                    records.append((age, layer, x, value, value + 2, value + 1,
                                    .2, .4, .3))
            columns = ["age_ma", "layer_id", "x", "temperature_top", "temperature_bottom",
                       "temperature_average", "reflectance_top", "reflectance_bottom",
                       "reflectance_average"]
            for inverse in (False, True):
                with self.subTest(inverse=inverse):
                    output = root / str(inverse)
                    fields = output / "forward" if inverse else output
                    fields.mkdir(parents=True)
                    pd.DataFrame(records, columns=columns).to_csv(fields / "fields.csv", index=False)
                    if inverse:
                        pd.DataFrame({"age_ma": [1., 0.], "basal_heat_flux_mw_m2": [90., 60.]}).to_csv(
                            output / "parameters.csv", index=False)
                    export_results(source, output, inverse)
                    for filename in ("Temperature.xlsx", "Sweeney_Burnham_1990_EASY_Ro.xlsx"):
                        pages = pd.read_excel(output / filename, sheet_name=None)
                        self.assertEqual(list(pages), ["Layer_7", "Layer_1"])
                        for frame in pages.values():
                            np.testing.assert_array_equal(frame.iloc[:, :5], geometry)
                            self.assertTrue(np.isfinite(frame.iloc[:, 5:]).all().all())
                            np.testing.assert_allclose(frame.iloc[:, 7],
                                                       .5 * (frame.iloc[:, 5] + frame.iloc[:, 6]))
                    if inverse:
                        flux = pd.read_excel(output / "Heat_Flow_Basal.xlsx", sheet_name=None)
                        for frame in flux.values():
                            np.testing.assert_array_equal(frame.iloc[:, :5], geometry)
                            np.testing.assert_array_equal(frame.iloc[:, 5:],
                                                           np.tile([60., 60., 90., 90.], (3, 1)).T)


if __name__ == "__main__":
    unittest.main()

"""Write calculated T, Ro and inverse basal flux on the original XLSX geometry."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from basin_data import _layer_sort_key


def export_results(data_dir: Path, directory: Path, inverse: bool = False) -> None:
    geometry = pd.read_excel(data_dir / "Depth.xlsx", sheet_name=None)
    layer_ids = {name: i + 1 for i, name in enumerate(sorted(geometry, key=_layer_sort_key))}
    forward = directory / "forward" if inverse else directory
    fields = pd.read_csv(forward / "fields.csv", float_precision="round_trip")
    fields = fields.set_index(["age_ma", "layer_id", "x"])
    outputs = [("Temperature.xlsx", "temperature", "Celsius"),
               ("Sweeney_Burnham_1990_EASY_Ro.xlsx", "reflectance", "%Ro")]
    if inverse:
        parameters = pd.read_csv(directory / "parameters.csv").set_index("age_ma")
        outputs.append(("Heat_Flow_Basal.xlsx", "flux", "mW/m^2"))
    for filename, quantity, unit in outputs:
        with pd.ExcelWriter(directory / filename, engine="openpyxl") as writer:
            for name, source in geometry.items():
                frame = source.iloc[:, :5].copy()
                if quantity == "flux":
                    q = frame.iloc[:, 1].map(parameters.basal_heat_flux_mw_m2).to_numpy()
                    values = np.column_stack((q, q, q))
                else:
                    index = pd.MultiIndex.from_arrays(
                        [frame.iloc[:, 1], np.full(len(frame), layer_ids[name]), frame.iloc[:, 0]])
                    columns = [f"{quantity}_{part}" for part in ("top", "bottom", "average")]
                    values = fields.loc[index, columns].to_numpy()
                frame[[f"{part} [{unit}]" for part in ("Top", "Bot", "Aver")]] = values
                frame.to_excel(writer, sheet_name=name, index=False)
        print(directory / filename)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="Original XLSX input directory")
    parser.add_argument("directory", type=Path, help="Calculation output directory")
    parser.add_argument("--inverse", action="store_true", help="Export forward/ fields and fitted q")
    args = parser.parse_args()
    export_results(args.data_dir, args.directory, args.inverse)


if __name__ == "__main__":
    main()

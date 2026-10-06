"""Convert organizer spreadsheets to the compact forward-solver input."""

from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np

from prepare_mesh_geometry import (
    sampled_layers,
    sampling_points,
)


def load_module(path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location("organizer_basin_data", path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def write_csv(path: Path, header: tuple[str, ...], rows: list[tuple[object, ...]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def nearest_values(
    source_x: np.ndarray, source_values: np.ndarray, target_x: np.ndarray
) -> np.ndarray:
    right = np.searchsorted(source_x, target_x, side="left")
    right = np.clip(right, 0, len(source_x) - 1)
    left = np.maximum(right - 1, 0)
    use_left = np.abs(target_x - source_x[left]) <= np.abs(source_x[right] - target_x)
    return source_values[np.where(use_left, left, right)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare compact forward-problem input")
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("cell_size", type=float)
    parser.add_argument(
        "--basin-data",
        type=Path,
        default=Path("tools/basin_data.py"),
    )
    arguments = parser.parse_args()

    module = load_module(arguments.basin_data)
    data = module.BasinData(arguments.data_dir)
    ages = tuple(reversed([float(age) for age in data.times]))

    arguments.output.mkdir(parents=True, exist_ok=True)
    layer_ids = {name: index + 1 for index, name in enumerate(data.layers)}

    configuration_rows: list[tuple[object, ...]] = []
    layer_rows: list[tuple[object, ...]] = []
    boundary_rows: list[tuple[object, ...]] = []
    for age in ages:
        snapshot = data.snapshot(age)
        geometry = sampled_layers(snapshot, sampling_points(snapshot, arguments.cell_size))
        configuration_rows.append((f"{age:.15g}",))

        for layer, sampled in zip(snapshot, geometry, strict=True):
            layer_id = layer_ids[str(layer["layer"])]
            source_x = np.asarray(layer["x"], dtype=np.float64)
            target_x = sampled["x"]
            lithotypes = nearest_values(
                source_x, np.asarray(layer["lith"], dtype=int), target_x
            )
            porosities = np.interp(
                target_x, source_x, np.asarray(layer["porosity"], dtype=np.float64)
            )
            for x, top, bottom, lithotype, porosity in zip(
                target_x,
                sampled["top"],
                sampled["bottom"],
                lithotypes,
                porosities,
                strict=True,
            ):
                layer_rows.append(
                    (
                        f"{age:.15g}",
                        layer_id,
                        layer["layer"],
                        f"{float(x):.15g}",
                        f"{float(top):.15g}",
                        f"{float(bottom):.15g}",
                        int(lithotype),
                        f"{0.01 * float(porosity):.15g}",
                    )
                )

        surface_x, surface_values = data.tsurf(age)
        for x, value in zip(surface_x, surface_values, strict=True):
            boundary_rows.append(
                (f"{age:.15g}", "surface_temperature", f"{float(x):.15g}",
                 f"{float(value) + 273.15:.15g}")
            )
        flow_x, flow_values = data.heat_flow(age)
        for x, value in zip(flow_x, flow_values, strict=True):
            boundary_rows.append(
                (f"{age:.15g}", "basal_heat_flux", f"{float(x):.15g}",
                 f"{1.0e-3 * float(value):.15g}")
            )

    write_csv(
        arguments.output / "configurations.csv",
        ("age_ma",),
        configuration_rows,
    )
    write_csv(
        arguments.output / "layers.csv",
        ("age_ma", "layer_id", "layer_name", "x", "z_top", "z_bottom",
         "lithotype", "porosity"),
        layer_rows,
    )
    write_csv(
        arguments.output / "boundaries.csv",
        ("age_ma", "kind", "x", "value_si"),
        boundary_rows,
    )

    lithotype_rows = []
    for code, values in sorted(data.lithotypes.items()):
        lithotype_rows.append(
            (
                code,
                values["name"],
                f"{values['rho_s']:.15g}",
                f"{values['lambda_20']:.15g}",
                f"{values['c20_J']:.15g}",
                f"{1.0e-6 * values['As']:.15g}",
            )
        )
    write_csv(
        arguments.output / "lithotypes.csv",
        ("code", "name", "solid_density", "conductivity_20", "specific_heat_20",
         "heat_production"),
        lithotype_rows,
    )

    kinetics = data.easy_ro
    kinetics_rows = [
        (
            f"{kinetics['A']:.15g}",
            f"{float(energy):.15g}",
            f"{float(weight):.15g}",
            f"{kinetics['Rg_J']:.15g}",
        )
        for energy, weight in zip(kinetics["E_J"], kinetics["f"], strict=True)
    ]
    write_csv(
        arguments.output / "kinetics.csv",
        ("preexponential", "activation_energy", "weight", "gas_constant"),
        kinetics_rows,
    )

    print(
        f"prepared={arguments.output} configurations={len(ages)} "
        f"layers={len(layer_ids)} cell_size={arguments.cell_size:g}"
    )


if __name__ == "__main__":
    main()

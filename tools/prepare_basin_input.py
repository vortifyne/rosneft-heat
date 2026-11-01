"""Convert organizer spreadsheets to the compact forward-solver input."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from basin_data import BasinData


def write_csv(path: Path, header: tuple[str, ...], rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare compact forward-problem input")
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()

    data = BasinData(arguments.data_dir)
    ages = tuple(float(age) for age in reversed(data.times))

    arguments.output.mkdir(parents=True, exist_ok=True)
    layer_ids = {name: index + 1 for index, name in enumerate(data.layers)}

    configuration_rows = []
    layer_rows = []
    boundary_rows = []
    for age in ages:
        snapshot = data.snapshot(age)

        for layer in snapshot:
            layer_id = layer_ids[str(layer["layer"])]
            for x, top, bottom, lithotype, porosity in zip(
                layer["x"],
                layer["z_top"],
                layer["z_bot"],
                layer["lith"],
                layer["porosity"],
                strict=True,
            ):
                layer_rows.append(
                    (
                        f"{age:.15g}",
                        layer_id,
                        layer["layer"],
                        f"{float(x):.17g}",
                        f"{float(top):.17g}",
                        f"{float(bottom):.17g}",
                        int(lithotype),
                        f"{0.01 * float(porosity):.17g}",
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
        ((f"{age:.15g}",) for age in ages),
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

    write_csv(
        arguments.output / "lithotypes.csv",
        ("code", "name", "solid_density", "conductivity_20", "specific_heat_20",
         "heat_production"),
        ((code, values["name"], f"{values['rho_s']:.15g}",
          f"{values['lambda_20']:.15g}", f"{values['c20_J']:.15g}",
          f"{1.0e-6 * values['As']:.15g}")
         for code, values in sorted(data.lithotypes.items())),
    )

    kinetics = data.easy_ro
    write_csv(
        arguments.output / "kinetics.csv",
        ("preexponential", "activation_energy", "weight", "gas_constant"),
        ((f"{kinetics['A']:.15g}", f"{float(energy):.15g}",
          f"{float(weight):.15g}", f"{kinetics['Rg_J']:.15g}")
         for energy, weight in zip(kinetics["E_J"], kinetics["f"], strict=True)),
    )

    print(f"prepared={arguments.output} configurations={len(ages)} layers={len(layer_ids)}")


if __name__ == "__main__":
    main()

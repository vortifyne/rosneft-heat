"""Run coupled space-time refinements and compare final column-grid fields."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from column_snapshots import read_snapshot


def compare_fields(coarse: dict, fine: dict) -> dict[str, float]:
    """Integrate differences of cellwise constants over same-layer intersections."""
    if coarse["kind"] != fine["kind"] or coarse["age"] != fine["age"]:
        raise ValueError("Snapshots must describe the same age and event phase")

    def columns(snapshot: dict) -> dict:
        result = {}
        for index, (column, layer) in enumerate(zip(snapshot["column"], snapshot["layer"])):
            result.setdefault(int(column), {}).setdefault(int(layer), []).append(index)
        for layers in result.values():
            for indices in layers.values():
                indices.sort(key=lambda index: snapshot["top"][index])
        return result

    coarse_columns, fine_columns = columns(coarse), columns(fine)
    area = error1 = error2 = maximum = 0.0
    left_column = right_column = 0
    while left_column + 1 < len(coarse["x"]) and right_column + 1 < len(fine["x"]):
        left = max(coarse["x"][left_column], fine["x"][right_column])
        right = min(coarse["x"][left_column + 1], fine["x"][right_column + 1])
        if right > left:
            coarse_layers = coarse_columns.get(left_column, {})
            fine_layers = fine_columns.get(right_column, {})
            for layer in coarse_layers.keys() & fine_layers.keys():
                coarse_cells, fine_cells = coarse_layers[layer], fine_layers[layer]
                i = j = 0
                while i < len(coarse_cells) and j < len(fine_cells):
                    first, second = coarse_cells[i], fine_cells[j]
                    top = max(coarse["top"][first], fine["top"][second])
                    bottom = min(coarse["bottom"][first], fine["bottom"][second])
                    if bottom > top:
                        weight = float((right - left) * (bottom - top))
                        difference = abs(float(coarse["temperature"][first] -
                                               fine["temperature"][second]))
                        area += weight
                        error1 += weight * difference
                        error2 += weight * difference * difference
                        maximum = max(maximum, difference)
                    if coarse["bottom"][first] <= fine["bottom"][second]:
                        i += 1
                    else:
                        j += 1
        if coarse["x"][left_column + 1] <= fine["x"][right_column + 1]:
            left_column += 1
        else:
            right_column += 1

    if area <= 0.0:
        raise ValueError("Snapshots have no overlapping cells in the same layers")

    def total_area(snapshot: dict) -> float:
        width = np.diff(snapshot["x"])[snapshot["column"]]
        return float(np.sum(width * (snapshot["bottom"] - snapshot["top"])))

    return dict(l1_kelvin=error1 / area, l2_kelvin=math.sqrt(error2 / area),
                linf_kelvin=maximum, overlap_area=area,
                coarse_coverage=area / total_area(coarse),
                fine_coverage=area / total_area(fine))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     epilog="Additional basin_forward_solver options follow --.")
    parser.add_argument("input", type=Path, help="Prepared CSV input directory")
    parser.add_argument("output", type=Path)
    parser.add_argument("--dx", type=float, default=100.0)
    parser.add_argument("--dz", type=float, default=100.0)
    parser.add_argument("--dt-ma", type=float, default=2.0)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--runner", type=Path,
                        default=Path(__file__).with_name("run_basin_forward.sh"))
    options = sys.argv[1:]
    separator = options.index("--") if "--" in options else len(options)
    arguments = parser.parse_args(options[:separator])
    solver_options = options[separator + 1:]
    if arguments.levels < 2 or any(not math.isfinite(value) or value <= 0.0
                                   for value in (arguments.dx, arguments.dz, arguments.dt_ma)):
        parser.error("At least two levels and finite positive mesh/time steps are required")
    arguments.output.mkdir(parents=True, exist_ok=True)
    cases = []
    snapshots = []
    for level in range(arguments.levels):
        factor = 2 ** level
        dx, dz, dt = arguments.dx / factor, arguments.dz / factor, arguments.dt_ma / factor
        directory = arguments.output / f"level_{level}"
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Refusing to mix runs in {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        command = [str(arguments.runner.resolve()), str(arguments.input.resolve()),
                   str(directory.resolve()), "--dx", str(dx), "--dz", str(dz),
                   "--dt-ma", str(dt), *solver_options]
        started = time.monotonic()
        with (directory / "run.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        elapsed = time.monotonic() - started
        paths = sorted(directory.glob("snapshot_*.cgrd"))
        if not paths:
            raise FileNotFoundError(f"No snapshots produced in {directory}")
        snapshot = read_snapshot(paths[-1])
        if not np.all(np.isfinite(snapshot["temperature"])):
            raise ValueError(f"Nonfinite temperatures in {paths[-1]}")
        snapshots.append(snapshot)
        cases.append(dict(level=level, dx=dx, dz=dz, dt_ma=dt,
                          cells=len(snapshot["temperature"]), elapsed_seconds=elapsed,
                          snapshot=str(paths[-1])))
        print(f"level={level} dx={dx:g} dz={dz:g} dt_ma={dt:g} "
              f"cells={cases[-1]['cells']} elapsed_seconds={elapsed:.3f}", flush=True)

    comparisons = []
    for level in range(len(snapshots) - 1):
        comparison = dict(coarse_level=level, fine_level=level + 1,
                          **compare_fields(snapshots[level], snapshots[level + 1]))
        comparisons.append(comparison)
        print(f"levels={level}/{level + 1} L2_K={comparison['l2_kelvin']:.6g} "
              f"Linf_K={comparison['linf_kelvin']:.6g} "
              f"coverage={comparison['coarse_coverage']:.6f}/{comparison['fine_coverage']:.6f}")
    result = dict(cases=cases, comparisons=comparisons, solver_options=solver_options)
    if len(comparisons) >= 2 and comparisons[-1]["l2_kelvin"] > 0.0:
        result["last_l2_ratio"] = comparisons[-2]["l2_kelvin"] / comparisons[-1]["l2_kelvin"]
    (arguments.output / "convergence.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

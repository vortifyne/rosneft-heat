"""Run and summarize a coupled space-time refinement series for data1."""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Case:
    name: str
    cell_size: float
    timestep_ma: float


@dataclass
class VtuField:
    points: np.ndarray
    cells: list[list[int]]
    centers: np.ndarray
    temperature: np.ndarray
    neighbors: list[list[int]]


def read_single_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 1:
        raise RuntimeError(f"Expected one data row in {path}")
    return rows[0]


def data_arrays(piece: ET.Element, parent: str) -> dict[str, list[str]]:
    node = piece.find(parent)
    if node is None:
        raise RuntimeError(f"VTU section {parent} is absent")
    return {
        array.get("Name", "_"): (array.text or "").split()
        for array in node.findall("DataArray")
    }


def read_vtu(path: Path) -> VtuField:
    piece = ET.parse(path).getroot().find(".//Piece")
    if piece is None:
        raise RuntimeError(f"VTU piece is absent in {path}")
    raw_points = data_arrays(piece, "Points")["_"]
    points = np.asarray(raw_points, dtype=np.float64).reshape((-1, 3))[:, :2]
    cell_arrays = data_arrays(piece, "Cells")
    connectivity = np.asarray(cell_arrays["connectivity"], dtype=np.int64)
    offsets = np.asarray(cell_arrays["offsets"], dtype=np.int64)
    cells: list[list[int]] = []
    start = 0
    for offset in offsets:
        cells.append(connectivity[start:offset].tolist())
        start = int(offset)
    temperature = np.asarray(
        data_arrays(piece, "CellData")["temperature"], dtype=np.float64
    )
    centers = np.asarray([points[cell].mean(axis=0) for cell in cells])
    neighbors: list[list[int]] = [[] for _ in cells]
    edges: dict[tuple[int, int], int] = {}
    for cell_index, cell in enumerate(cells):
        for index, first in enumerate(cell):
            second = cell[(index + 1) % len(cell)]
            key = tuple(sorted((first, second)))
            if key in edges:
                neighbor = edges[key]
                neighbors[cell_index].append(neighbor)
                neighbors[neighbor].append(cell_index)
            else:
                edges[key] = cell_index
    return VtuField(points, cells, centers, temperature, neighbors)


def contains(polygon: np.ndarray, point: np.ndarray) -> bool:
    cross = []
    for index in range(len(polygon)):
        edge = polygon[(index + 1) % len(polygon)] - polygon[index]
        offset = point - polygon[index]
        cross.append(edge[0] * offset[1] - edge[1] * offset[0])
    scale = max(1.0, float(np.max(np.abs(polygon))))
    tolerance = 1.0e-10 * scale * scale
    return min(cross) >= -tolerance or max(cross) <= tolerance


def sample(field: VtuField, point: np.ndarray) -> float:
    donor = None
    for index, cell in enumerate(field.cells):
        polygon = field.points[cell]
        if np.all(point >= polygon.min(axis=0)) and np.all(point <= polygon.max(axis=0)):
            if contains(polygon, point):
                donor = index
                break
    if donor is None:
        raise RuntimeError(f"Comparison point {point.tolist()} is outside the mesh")
    adjacent = field.neighbors[donor]
    if not adjacent:
        return float(field.temperature[donor])
    displacement = field.centers[adjacent] - field.centers[donor]
    difference = field.temperature[adjacent] - field.temperature[donor]
    gradient, *_ = np.linalg.lstsq(displacement, difference, rcond=None)
    reconstructed = float(
        field.temperature[donor] + gradient @ (point - field.centers[donor])
    )
    stencil = np.concatenate(
        ([field.temperature[donor]], field.temperature[adjacent])
    )
    return float(np.clip(reconstructed, stencil.min(), stencil.max()))


def select_points(field: VtuField, count: int = 20) -> np.ndarray:
    minimum = field.centers.min(axis=0)
    extent = np.maximum(field.centers.max(axis=0) - minimum, 1.0)
    normalized = (field.centers - minimum) / extent
    selected = [int(np.argmin(np.linalg.norm(normalized - 0.5, axis=1)))]
    distance = np.linalg.norm(normalized - normalized[selected[0]], axis=1)
    while len(selected) < min(count, len(field.cells)):
        candidate = int(np.argmax(distance))
        selected.append(candidate)
        distance = np.minimum(
            distance, np.linalg.norm(normalized - normalized[candidate], axis=1)
        )
    return field.centers[selected]


def final_vtu(directory: Path) -> Path:
    candidates = sorted(directory.glob("state_*_after_transition.vtu"))
    if not candidates:
        raise RuntimeError(f"No final state found in {directory}")
    return candidates[-1]


def run_case(solver: Path, shared_input: Path, output: Path, case: Case) -> float:
    if output.exists():
        raise RuntimeError(f"Output directory already exists: {output}")
    command = [
        str(solver),
        str(shared_input),
        str(output),
        "--cell-size",
        f"{case.cell_size:.15g}",
        "--dt-ma",
        f"{case.timestep_ma:.15g}",
    ]
    start = time.monotonic()
    subprocess.run(command, check=True, timeout=15 * 60)
    return time.monotonic() - start


def write_case_summary(output: Path, cases: list[tuple[Case, Path, float]]) -> None:
    columns = [
        "case",
        "cell_size_m",
        "timestep_ma",
        "wall_seconds_external",
        "accepted_steps",
        "nonlinear_iterations",
        "cells_min",
        "cells_max",
        "global_energy_balance",
        "cell_thickness_min",
        "cell_elongation_max",
    ]
    with (output / "cases.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=columns)
        writer.writeheader()
        for case, directory, wall_seconds in cases:
            summary = read_single_row(directory / "summary.csv")
            writer.writerow(
                {
                    "case": case.name,
                    "cell_size_m": case.cell_size,
                    "timestep_ma": case.timestep_ma,
                    "wall_seconds_external": f"{wall_seconds:.9g}",
                    **{column: summary[column] for column in columns[4:]},
                }
            )


def write_temperature_sequence(
    output: Path, cases: list[tuple[Case, Path, float]]
) -> None:
    cases = sorted(cases, key=lambda item: item[0].cell_size, reverse=True)
    fields = [read_vtu(final_vtu(directory)) for _, directory, _ in cases]
    points = select_points(fields[0])
    temperature_columns = [f"temperature_{case.name}" for case, _, _ in cases]
    difference_columns = [
        f"difference_{first[0].name}_to_{second[0].name}"
        for first, second in zip(cases, cases[1:])
    ]
    with (output / "temperature-sequence.csv").open(
        "w", newline="", encoding="utf-8"
    ) as target:
        writer = csv.writer(target)
        writer.writerow(("point", "x", "z", *temperature_columns, *difference_columns))
        for index, point in enumerate(points):
            temperatures = [sample(field, point) for field in fields]
            differences = [
                abs(first - second)
                for first, second in zip(temperatures, temperatures[1:])
            ]
            writer.writerow((index, *point, *temperatures, *differences))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full data1 space-time convergence series"
    )
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--solver", type=Path, default=Path("build/release/heat_forward"))
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise RuntimeError(f"Output directory already exists: {arguments.output}")
    arguments.output.mkdir(parents=True)
    shared_input = arguments.output / "input"
    subprocess.run(
        [
            sys.executable,
            "tools/prepare_forward_data.py",
            str(arguments.data),
            str(shared_input),
            "25",
        ],
        check=True,
    )

    coarse = Case("h100-dt1", 100.0, 1.0)
    medium = Case("h50-dt0.25", 50.0, 0.25)
    completed: list[tuple[Case, Path, float]] = []
    for case in (coarse, medium):
        directory = arguments.output / case.name
        completed.append(
            (case, directory, run_case(arguments.solver, shared_input, directory, case))
        )

    predicted_fine_seconds = completed[-1][2] ** 2 / max(completed[0][2], 1.0e-9)
    if predicted_fine_seconds <= 15 * 60:
        fine = Case("h25-dt0.0625", 25.0, 0.0625)
        directory = arguments.output / fine.name
        fine_result = (
            fine,
            directory,
            run_case(arguments.solver, shared_input, directory, fine),
        )
        completed.append(fine_result)
        middle = Case("h35.3553-dt0.125", math.sqrt(1250.0), 0.125)
        directory = arguments.output / middle.name
        middle_result = (
            middle,
            directory,
            run_case(arguments.solver, shared_input, directory, middle),
        )
        completed.append(middle_result)
    else:
        middle = Case("h70.7107-dt0.5", math.sqrt(5000.0), 0.5)
        directory = arguments.output / middle.name
        middle_result = (
            middle,
            directory,
            run_case(arguments.solver, shared_input, directory, middle),
        )
        completed.append(middle_result)

    write_case_summary(arguments.output, completed)
    write_temperature_sequence(arguments.output, completed)
    print(f"verification={arguments.output} cases={len(completed)}")


if __name__ == "__main__":
    main()

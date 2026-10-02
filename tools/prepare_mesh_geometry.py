"""Prepare TQMesh layer contours for one basin configuration."""

from __future__ import annotations

import argparse
import csv
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TOP = 1
BOTTOM = 2
LEFT = 3
RIGHT = 4


@dataclass(frozen=True)
class Region:
    layer_id: int
    vertices: tuple[tuple[float, float], ...]
    edge_kinds: tuple[int, ...]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare geological layer contours for TQMesh"
    )
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("time", help="Configuration time or 'all'")
    parser.add_argument("step", type=float, help="Boundary point spacing")
    parser.add_argument("output", type=Path)
    parser.add_argument("--plot", type=Path)
    parser.add_argument(
        "--basin-data",
        type=Path,
        default=Path("local/forward_problem_data/basin_data.py"),
    )
    return parser.parse_args()


def load_basin_module(path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location("organizer_basin_data", path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def sampling_points(snapshot: list[dict[str, object]], step: float) -> np.ndarray:
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("Boundary point spacing must be finite and positive")
    x_min = min(float(np.asarray(layer["x"])[0]) for layer in snapshot)
    x_max = max(float(np.asarray(layer["x"])[-1]) for layer in snapshot)
    regular = np.arange(x_min, x_max + 0.5 * step, step, dtype=np.float64)
    endpoints = np.asarray(
        [coordinate for layer in snapshot for coordinate in (layer["x"][0], layer["x"][-1])],
        dtype=np.float64,
    )
    geometry_events: list[float] = []
    for layer in snapshot:
        x = np.asarray(layer["x"], dtype=np.float64)
        thickness = np.asarray(layer["z_bot"], dtype=np.float64) - np.asarray(
            layer["z_top"], dtype=np.float64
        )
        tolerance = 1.0e-10 * max(1.0, float(np.max(np.abs(thickness))))
        for index in np.flatnonzero(thickness <= tolerance):
            first = max(0, int(index) - 1)
            last = min(len(x), int(index) + 2)
            geometry_events.extend(x[first:last])
    return np.unique(
        np.clip(
            np.concatenate((regular, endpoints, geometry_events, [x_max])),
            x_min,
            x_max,
        )
    )


def sampled_layers(
    snapshot: list[dict[str, object]], points: np.ndarray
) -> list[dict[str, np.ndarray]]:
    result: list[dict[str, np.ndarray]] = []
    for layer in snapshot:
        source_x = np.asarray(layer["x"], dtype=np.float64)
        x = points[(points >= source_x[0]) & (points <= source_x[-1])]
        result.append(
            {
                "x": x,
                "top": np.interp(x, source_x, np.asarray(layer["z_top"], dtype=np.float64)),
                "bottom": np.interp(
                    x, source_x, np.asarray(layer["z_bot"], dtype=np.float64)
                ),
            }
        )

    # The same numerical points must describe both sides of a shared boundary.
    for upper, lower in zip(result, result[1:]):
        common, upper_indices, lower_indices = np.intersect1d(
            upper["x"], lower["x"], assume_unique=True, return_indices=True
        )
        if common.size == 0:
            continue
        shared = 0.5 * (
            upper["bottom"][upper_indices] + lower["top"][lower_indices]
        )
        upper["bottom"][upper_indices] = shared
        lower["top"][lower_indices] = shared
    return result


def same_point(first: tuple[float, float], second: tuple[float, float]) -> bool:
    scale = max(1.0, *(abs(value) for value in (*first, *second)))
    return max(abs(first[0] - second[0]), abs(first[1] - second[1])) <= 1.0e-12 * scale


def make_region(layer: dict[str, np.ndarray], layer_id: int) -> Region | None:
    top = list(zip(layer["x"], layer["top"], strict=True))
    bottom = list(reversed(list(zip(layer["x"], layer["bottom"], strict=True))))
    tagged = [((float(x), float(z)), TOP) for x, z in top]
    tagged += [((float(x), float(z)), BOTTOM) for x, z in bottom]

    vertices: list[tuple[float, float]] = []
    tags: list[int] = []
    for point, tag in tagged:
        if vertices and same_point(vertices[-1], point):
            continue
        vertices.append(point)
        tags.append(tag)
    if len(vertices) > 1 and same_point(vertices[0], vertices[-1]):
        vertices.pop()
        tags.pop()
    if len(vertices) < 3:
        return None

    edge_kinds: list[int] = []
    for index in range(len(vertices)):
        next_index = (index + 1) % len(vertices)
        if tags[index] == tags[next_index]:
            edge_kinds.append(tags[index])
        elif vertices[index][0] < vertices[next_index][0]:
            edge_kinds.append(RIGHT)
        else:
            edge_kinds.append(LEFT)
    return Region(layer_id, tuple(vertices), tuple(edge_kinds))


def split_at_pinches(layer: dict[str, np.ndarray]) -> list[dict[str, np.ndarray]]:
    thickness = layer["bottom"] - layer["top"]
    tolerance = 1.0e-10 * max(1.0, float(np.max(np.abs(thickness))))
    cuts = [0]
    cuts.extend(
        index
        for index in range(1, len(thickness) - 1)
        if thickness[index] <= tolerance
    )
    cuts.append(len(thickness) - 1)

    pieces: list[dict[str, np.ndarray]] = []
    for first, last in zip(cuts, cuts[1:]):
        if last <= first:
            continue
        section = slice(first, last + 1)
        if np.max(thickness[section]) <= tolerance:
            continue
        pieces.append({name: values[section] for name, values in layer.items()})
    return pieces


def extract_regions(
    snapshot: list[dict[str, object]],
    step: float,
    layer_ids: dict[str, int] | None = None,
) -> list[Region]:
    points = sampling_points(snapshot, step)
    layers = sampled_layers(snapshot, points)
    regions = [
        make_region(
            piece,
            layer_ids[str(snapshot[index]["layer"])] if layer_ids else index + 1,
        )
        for index, layer in enumerate(layers)
        for piece in split_at_pinches(layer)
    ]
    return [region for region in regions if region is not None]


def write_regions(path: Path, regions: list[Region]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(("region_id", "layer_id", "x", "z", "edge_kind"))
        for region_id, region in enumerate(regions):
            for (x, z), kind in zip(region.vertices, region.edge_kinds, strict=True):
                writer.writerow((region_id, region.layer_id, f"{x:.15g}", f"{z:.15g}", kind))


def draw_regions(path: Path, regions: list[Region], time: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(12, 6), constrained_layout=True)
    for region in regions:
        coordinates = np.asarray(region.vertices + (region.vertices[0],))
        axes.plot(coordinates[:, 0], coordinates[:, 1], linewidth=0.7)
    axes.invert_yaxis()
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel("x")
    axes.set_ylabel("depth")
    axes.set_title(f"Layer contours, {time:g} Ma")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    arguments = parse_arguments()
    module = load_basin_module(arguments.basin_data)
    data = module.BasinData(arguments.data_dir)
    times = data.times if arguments.time == "all" else [float(arguments.time)]
    for time in times:
        regions = extract_regions(data.snapshot(time), arguments.step)
        if not regions:
            raise RuntimeError(f"The configuration at {time:g} contains no layers")
        output = arguments.output / f"{time:g}.csv" if len(times) > 1 else arguments.output
        write_regions(output, regions)
        if arguments.plot is not None:
            plot = arguments.plot / f"{time:g}.png" if len(times) > 1 else arguments.plot
            draw_regions(plot, regions, time)
        print(
            f"time={time:g} layers={len(data.layers_at(time))} regions={len(regions)} "
            f"boundary_vertices={sum(len(region.vertices) for region in regions)}"
        )


if __name__ == "__main__":
    main()

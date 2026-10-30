"""Browse column-grid snapshots with a temperature or EASY%Ro slider."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.widgets import Slider
import numpy as np

from column_snapshots import read_snapshot


def polygons(snapshot: dict) -> np.ndarray:
    left = snapshot["x"][snapshot["column"]]
    right = snapshot["x"][snapshot["column"] + 1]
    top, bottom = snapshot["top"], snapshot["bottom"]
    return np.stack((np.column_stack((left, top)),
                     np.column_stack((right, top)),
                     np.column_stack((right, bottom)),
                     np.column_stack((left, bottom))), axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--field", choices=("temperature", "reflectance", "layer"),
                        default="temperature")
    arguments = parser.parse_args()
    paths = sorted(arguments.directory.glob("snapshot_*.cgrd"))
    if not paths:
        parser.error("No snapshot_*.cgrd files found")
    snapshots = [read_snapshot(path) for path in paths]
    key = arguments.field
    if key == "layer":
        data = [snapshot["layer"] for snapshot in snapshots]
    else:
        data = [snapshot[key] for snapshot in snapshots]
    minimum = min(float(np.min(values)) for values in data)
    maximum = max(float(np.max(values)) for values in data)
    if maximum == minimum:
        maximum = minimum + 1.0

    figure, axis = plt.subplots(figsize=(12, 7))
    figure.subplots_adjust(bottom=0.17)
    collection = PolyCollection(polygons(snapshots[0]), array=data[0],
                                cmap="tab20" if key == "layer" else "inferno",
                                edgecolors="none", clim=(minimum, maximum))
    axis.add_collection(collection)
    figure.colorbar(collection, ax=axis, label=key)
    left = min(float(snapshot["x"][0]) for snapshot in snapshots)
    right = max(float(snapshot["x"][-1]) for snapshot in snapshots)
    top = min(float(np.min(snapshot["top"])) for snapshot in snapshots)
    bottom = max(float(np.max(snapshot["bottom"])) for snapshot in snapshots)
    axis.set_xlim(left, right)
    axis.set_ylim(bottom, top)
    axis.set_xlabel("x, m")
    axis.set_ylabel("depth, m")

    def title(index: int) -> str:
        snapshot = snapshots[index]
        phase = ("initial", "before event", "after event")[snapshot["kind"]]
        return f"{snapshot['age']:g} Ma — {phase} — {snapshot['path'].name}"

    axis.set_title(title(0))
    slider_axis = figure.add_axes((0.12, 0.05, 0.75, 0.035))
    slider = Slider(slider_axis, "snapshot", 0, len(snapshots) - 1,
                    valinit=0, valstep=1)

    def update(position: float) -> None:
        index = int(position)
        collection.set_verts(polygons(snapshots[index]))
        collection.set_array(data[index])
        axis.set_title(title(index))
        figure.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


if __name__ == "__main__":
    main()

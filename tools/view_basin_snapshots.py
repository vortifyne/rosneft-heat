"""Browse basin snapshots with a temperature or EASY%Ro slider."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.widgets import Slider
import numpy as np
import pandas as pd

from basin_snapshot import cell_reflectance, read_snapshot


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
    parser.add_argument("directory", type=Path, help="Calculation output directory")
    parser.add_argument("--field", choices=("temperature", "reflectance", "layer"),
                        default="temperature")
    parser.add_argument("--observations", type=Path, help="CSV with well coordinates and depths")
    parser.add_argument("--save", type=Path, help="Save the plot instead of opening a window")
    arguments = parser.parse_args()
    paths = sorted((arguments.directory / "snapshots").glob("snapshot_*.cgrd"),
                   key=lambda path: tuple(map(int, re.findall(r"\d+", path.stem))))
    if not paths:
        parser.error(f"No snapshot_*.cgrd files found in {arguments.directory / 'snapshots'}")
    snapshots = [read_snapshot(path) for path in paths]
    key = arguments.field
    initial = len(snapshots) - 1 if arguments.observations else 0
    data = [cell_reflectance(snapshot) if key == "reflectance" else snapshot[key]
            for snapshot in snapshots]
    minimum = min(float(np.min(values)) for values in data)
    maximum = max(float(np.max(values)) for values in data)
    if maximum == minimum:
        maximum = minimum + 1.0

    figure, axis = plt.subplots(figsize=(12, 7))
    figure.subplots_adjust(bottom=0.17)
    collection = PolyCollection(polygons(snapshots[initial]), array=data[initial],
                                cmap="tab20" if key == "layer" else "inferno",
                                edgecolors="none", clim=(minimum, maximum))
    axis.add_collection(collection)
    labels = {"temperature": "Temperature, K", "reflectance": "Ro, %", "layer": "Layer"}
    figure.colorbar(collection, ax=axis, label=labels[key])
    left = min(float(snapshot["x"][0]) for snapshot in snapshots)
    right = max(float(snapshot["x"][-1]) for snapshot in snapshots)
    top = min(float(np.min(snapshot["top"])) for snapshot in snapshots)
    bottom = max(float(np.max(snapshot["bottom"])) for snapshot in snapshots)
    axis.set_xlim(left, right)
    axis.set_ylim(bottom, top)
    axis.set_xlabel("x, m")
    axis.set_ylabel("depth, m")
    well_artists = []
    if arguments.observations:
        points = pd.read_csv(arguments.observations).drop_duplicates(["well", "x", "z"])
        for well, values in points.groupby("well", sort=False):
            values = values.sort_values("z")
            well_artists.extend(axis.plot(values.x, values.z, ".-", color="cyan",
                                           markersize=3, linewidth=0.8))
            well_artists.append(axis.text(values.x.iloc[0], values.z.iloc[0], str(well),
                                           fontsize=9, color="black",
                                           bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"}))

    def title(index: int) -> str:
        snapshot = snapshots[index]
        phase = ("initial", "before event", "after event", "timestep")[snapshot["kind"]]
        return f"{snapshot['age']:g} Ma — {phase} — {snapshot['path'].name}"

    axis.set_title(title(initial))

    def update(position: float) -> None:
        index = int(position)
        collection.set_verts(polygons(snapshots[index]))
        collection.set_array(data[index])
        axis.set_title(title(index))
        for artist in well_artists:
            artist.set_visible(snapshots[index]["age"] == snapshots[-1]["age"])
        figure.canvas.draw_idle()

    if len(snapshots) > 1:
        slider_axis = figure.add_axes((0.12, 0.05, 0.75, 0.035))
        slider = Slider(slider_axis, "snapshot", 0, len(snapshots) - 1,
                        valinit=initial, valstep=1)
        slider.on_changed(update)
    if arguments.save:
        arguments.save.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(arguments.save, dpi=160)
        plt.close(figure)
    else:
        plt.show()


if __name__ == "__main__":
    main()

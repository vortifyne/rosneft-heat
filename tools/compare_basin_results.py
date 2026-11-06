"""Compare T/Ro XLSX files or two directories containing organizer-format results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PARTS = ("top", "bottom", "average")
FILES = ("Temperature.xlsx", "Sweeney_Burnham_1990_EASY_Ro.xlsx")


def read_difference(reference: Path, calculated: Path) -> tuple[str, str, pd.DataFrame]:
    first = pd.read_excel(reference, sheet_name=None)
    second = pd.read_excel(calculated, sheet_name=None)
    if set(first) != set(second):
        raise ValueError("Layer sheets do not match")
    header = str(next(iter(first.values())).columns[5])
    if "Celsius" in header:
        quantity, unit = "temperature", "°C"
    elif "%Ro" in header:
        quantity, unit = "reflectance", "%Ro"
    else:
        raise ValueError(f"Unsupported result units: {header}")
    frames = []
    for layer, a in first.items():
        b = second[layer]
        if not np.array_equal(a.iloc[:, :5].to_numpy(), b.iloc[:, :5].to_numpy()):
            raise ValueError(f"Geometry, ages or row order do not match: {layer}")
        if list(a.columns[5:8]) != list(b.columns[5:8]):
            raise ValueError(f"Field columns or units do not match: {layer}")
        if not np.isfinite(a.iloc[:, 5:8]).all().all() or not np.isfinite(b.iloc[:, 5:8]).all().all():
            raise ValueError(f"Nonfinite field values: {layer}")
        frame = pd.DataFrame({"layer": layer, "x": a.iloc[:, 0], "age_ma": a.iloc[:, 1],
                              "z_top": a.iloc[:, 2], "z_bottom": a.iloc[:, 3],
                              "z_average": .5 * (a.iloc[:, 2] + a.iloc[:, 3])})
        for j, part in enumerate(PARTS, 5):
            frame[f"reference_{part}"] = a.iloc[:, j].to_numpy()
            frame[f"error_{part}"] = b.iloc[:, j].to_numpy() - a.iloc[:, j].to_numpy()
        frames.append(frame)
    return quantity, unit, pd.concat(frames, ignore_index=True)


def error_metrics(frame: pd.DataFrame) -> list[dict]:
    result = []
    for part in PARTS:
        error = frame[f"error_{part}"].to_numpy()
        reference = frame[f"reference_{part}"].to_numpy()
        maximum = int(np.argmax(np.abs(error)))
        row = frame.iloc[maximum]
        norm = float(np.linalg.norm(reference))
        result.append(dict(component=part, count=len(error), max_abs=float(abs(error[maximum])),
                           rmse=float(np.sqrt(np.mean(error * error))),
                           relative_l2_percent=100. * float(np.linalg.norm(error)) / norm if norm else np.nan,
                           layer=row.layer, age_ma=float(row.age_ma), x=float(row.x),
                           z=float(row[f"z_{part}"])))
    return result


def plot_difference(frame: pd.DataFrame, quantity: str, unit: str, age: float):
    selected = frame[frame.age_ma == age]
    if selected.empty:
        raise ValueError(f"No configuration at age {age:g} Ma")
    limit = float(np.max(np.abs(selected[[f"error_{p}" for p in PARTS]].to_numpy()))) or 1.
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True,
                                layout="constrained")
    for axis, part, title in zip(axes, PARTS, ("Top", "Bot", "Aver")):
        error = selected[f"error_{part}"].to_numpy()
        depth = selected[f"z_{part}"].to_numpy()
        cloud = axis.scatter(selected.x, depth, c=error, s=6, cmap="coolwarm",
                             vmin=-limit, vmax=limit)
        maximum = int(np.argmax(np.abs(error)))
        axis.plot(selected.x.iloc[maximum], depth[maximum], "kx", markersize=8)
        axis.set_title(f"{title}: max |error| = {abs(error[maximum]):.4g} {unit}")
        axis.set_xlabel("x, m")
    axes[0].set_ylabel("depth, m")
    axes[0].invert_yaxis()
    figure.colorbar(cloud, ax=axes, label=f"Calculated − reference, {unit}")
    figure.suptitle(f"{quantity.capitalize()}, age {age:g} Ma")
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="Reference XLSX file or result directory")
    parser.add_argument("calculated", type=Path, help="Calculated XLSX file or result directory")
    parser.add_argument("--age", type=float, default=0., help="Plot age in Ma (default: 0)")
    parser.add_argument("--save", type=Path, help="Save PNG plots and summary.csv instead of opening windows")
    args = parser.parse_args()
    if args.reference.is_dir() and args.calculated.is_dir():
        pairs = [(args.reference / name, args.calculated / name) for name in FILES
                 if (args.reference / name).exists() and (args.calculated / name).exists()]
    else:
        pairs = [(args.reference, args.calculated)]
    if not pairs:
        parser.error("No matching T or Ro workbooks")
    if args.save:
        args.save.mkdir(parents=True, exist_ok=True)
    summary = []
    try:
        for reference, calculated in pairs:
            quantity, unit, frame = read_difference(reference, calculated)
            for metric in error_metrics(frame):
                summary.append(dict(field=quantity, unit=unit, **metric))
                print(f"{quantity} {metric['component']}: max={metric['max_abs']:.6g} {unit}; "
                      f"RMSE={metric['rmse']:.6g} {unit}; L2 relative={metric['relative_l2_percent']:.4g}%; "
                      f"at {metric['layer']}, age={metric['age_ma']:g} Ma, "
                      f"x={metric['x']:g} m, z={metric['z']:g} m")
            figure = plot_difference(frame, quantity, unit, args.age)
            if args.save:
                figure.savefig(args.save / f"{quantity}_age_{args.age:g}.png", dpi=160)
                plt.close(figure)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    if args.save:
        pd.DataFrame(summary).to_csv(args.save / "summary.csv", index=False)
    else:
        plt.show()


if __name__ == "__main__":
    main()

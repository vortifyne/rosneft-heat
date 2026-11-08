"""Plot basal heat flux, well observations and inverse-solver progress."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_flux(directory: Path):
    figure, axis = plt.subplots(figsize=(9, 4), layout="constrained")
    for filename, label in (("initial_parameters.csv", "Initial"),
                            ("parameters.csv", "Found")):
        path = directory / filename
        if path.exists():
            data = pd.read_csv(path)
            axis.plot(data.age_ma, data.basal_heat_flux_mw_m2, ".-", label=label)
    axis.invert_xaxis()
    axis.set(xlabel="Age, Ma", ylabel="Basal heat flux, mW/m²")
    axis.grid(alpha=0.25)
    axis.legend()
    return figure


def plot_wells(data: pd.DataFrame):
    wells = data.well.unique()
    profiles, axes = plt.subplots(2, len(wells), squeeze=False, figsize=(3.3 * len(wells), 9),
                                  layout="constrained")
    residuals, residual_axes = plt.subplots(2, len(wells), squeeze=False,
                                           figsize=(3.3 * len(wells), 9), layout="constrained")
    for row, (kind, label) in enumerate((("temperature", "Temperature, °C"),
                                        ("reflectance", "Ro, %"))):
        for column, well in enumerate(wells):
            points = data[(data.kind == kind) & (data.well == well)].sort_values("z")
            inside = int((points.normalized_residual.abs() <= 1.0).sum())
            axis = axes[row, column]
            axis.fill_betweenx(points.z, points.observed - points.error,
                               points.observed + points.error, color="0.75", alpha=0.5,
                               label="Measurement errors")
            axis.plot(points.observed, points.z, "k.", markersize=3, label="Measured")
            axis.plot(points.calculated, points.z, color="tab:blue", label="Calculated")
            axis.set(xlabel=label, title=f"{well}: {inside}/{len(points)}")

            axis = residual_axes[row, column]
            percent = 100.0 / points.observed.abs()
            bound = percent * points.error
            axis.fill_betweenx(points.z, -bound, bound, color="#dcebd6", label="Measurement errors")
            axis.axvline(0.0, color="0.6", linewidth=0.7)
            axis.plot(percent * (points.calculated - points.observed), points.z,
                      ".-", markersize=3, color="tab:blue", label="Residual")
            axis.set(xlabel="Relative residual, %", title=f"{well}: {inside}/{len(points)}")
            for axis in (axes[row, column], residual_axes[row, column]):
                axis.invert_yaxis()
                axis.grid(alpha=0.25)
                if column == 0:
                    axis.set_ylabel(("Temperature" if row == 0 else "Ro") + "; depth z, m")
    for figure, grid in ((profiles, axes), (residuals, residual_axes)):
        handles, labels = grid[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="outside lower center", ncol=len(labels))
        figure.suptitle("Counts: observations within measurement errors")
    return {"profiles": profiles, "residuals": residuals}


def plot_progress(data: pd.DataFrame):
    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 6), layout="constrained")
    axes[0].semilogy(data.evaluation, data.objective, ".-", markersize=3,
                     color="0.65", label="Evaluated objective")
    axes[0].semilogy(data.evaluation, np.minimum.accumulate(data.objective),
                     ".-", markersize=3, color="tab:blue", label="Minimum evaluated")
    axes[0].set_ylabel("Sum of squared residuals")
    axes[0].legend()
    axes[1].plot(data.evaluation, data.max_normalized_residual, ".-", markersize=3, color="tab:blue")
    axes[1].axhline(1.0, color="tab:green", linestyle="--", label="Within measurement errors")
    axes[1].set(xlabel="Forward evaluation", ylabel="Maximum normalized residual")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Inverse calculation output directory")
    parser.add_argument("--save", type=Path, help="Save PNG figures here instead of opening windows")
    arguments = parser.parse_args()
    figures = {"heat_flux": plot_flux(arguments.directory)}
    figures.update(plot_wells(pd.read_csv(arguments.directory / "observations.csv")))
    figures["convergence"] = plot_progress(pd.read_csv(arguments.directory / "progress.csv"))
    if arguments.save:
        arguments.save.mkdir(parents=True, exist_ok=True)
        for name, figure in figures.items():
            figure.savefig(arguments.save / f"{name}.png", dpi=160)
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()

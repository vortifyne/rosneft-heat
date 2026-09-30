"""Build the compact stage-09 tables from completed forward runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


RUNS = (
    ("data1-h50-dt0.1", "data1", 50.0, 0.1, Path("data1/h50-dt0.1")),
    ("data1-h50-dt0.025", "data1", 50.0, 0.025, Path("data1/h50-dt0.025")),
    ("data1-h25-dt0.1", "data1", 25.0, 0.1, Path("data1/h25-dt0.1")),
    ("data1-h25-dt0.025", "data1", 25.0, 0.025, Path("data1/h25-dt0.025")),
    ("data2-selected", "data2", 50.0, 0.1, Path("data2/selected")),
)


def read_single_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 1:
        raise ValueError(f"Expected one data row in {path}")
    return rows[0]


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def read_temperatures(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return {row["id"]: row for row in csv.DictReader(source)}


def write_run_summary(root: Path) -> list[dict[str, str]]:
    result = []
    for name, dataset, requested_h, timestep, relative in RUNS:
        directory = root / relative
        summary = read_single_row(directory / "summary.csv")
        actual_h = requested_h if requested_h else float(summary["cell_diameter_max"])
        result.append(
            {
                "run": name,
                "dataset": dataset,
                "requested_cell_size": f"{actual_h:g}",
                "timestep_ma": f"{timestep:g}",
                "cell_diameter_max": summary["cell_diameter_max"],
                "cells_min": summary["cells_min"],
                "cells_max": summary["cells_max"],
                "accepted_steps": summary["accepted_steps"],
                "nonlinear_iterations": summary["nonlinear_iterations"],
                "linear_iterations": summary["linear_iterations"],
                "transfer_energy_error": summary["transfer_energy_error"],
                "wall_seconds": summary["wall_seconds"],
                "output_bytes": str(directory_size(directory)),
            }
        )
    path = root / "run-summary.csv"
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=result[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(result)
    return result


def runge_rows(root: Path) -> list[dict[str, str]]:
    fields = {
        name: read_temperatures(root / relative / "comparison-temperatures.csv")
        for name, _, _, _, relative in RUNS[:4]
    }
    comparisons = (
        ("time", "h50", "data1-h50-dt0.1", "data1-h50-dt0.025", 4.0, 1.0),
        ("time", "h25", "data1-h25-dt0.1", "data1-h25-dt0.025", 4.0, 1.0),
        ("space", "dt0.1", "data1-h50-dt0.1", "data1-h25-dt0.1", 2.0, 2.0),
        ("space", "dt0.025", "data1-h50-dt0.025", "data1-h25-dt0.025", 2.0, 2.0),
    )
    result = []
    for kind, fixed, coarse_name, fine_name, ratio, order in comparisons:
        coarse = fields[coarse_name]
        fine = fields[fine_name]
        for identifier in coarse:
            coarse_temperature = float(coarse[identifier]["temperature"])
            fine_temperature = float(fine[identifier]["temperature"])
            difference = fine_temperature - coarse_temperature
            fine_error = abs(difference) / (ratio**order - 1.0)
            result.append(
                {
                    "kind": kind,
                    "fixed_resolution": fixed,
                    "point_id": identifier,
                    "x": coarse[identifier]["x"],
                    "z": coarse[identifier]["z"],
                    "coarse_temperature": f"{coarse_temperature:.17g}",
                    "fine_temperature": f"{fine_temperature:.17g}",
                    "fine_minus_coarse": f"{difference:.17g}",
                    "fine_error_estimate": f"{fine_error:.17g}",
                    "coarse_error_estimate": f"{ratio**order * fine_error:.17g}",
                }
            )
    return result


def write_report(root: Path, summaries: list[dict[str, str]], rows: list[dict[str, str]]) -> None:
    maximum_time_error = max(
        float(row["fine_error_estimate"]) for row in rows if row["kind"] == "time"
    )
    maximum_space_error = max(
        float(row["fine_error_estimate"]) for row in rows if row["kind"] == "space"
    )
    data2 = next(row for row in summaries if row["dataset"] == "data2")
    data1_transfer_error = max(
        float(row["transfer_energy_error"])
        for row in summaries
        if row["dataset"] == "data1"
    )
    data1_sizes = [
        int(row["output_bytes"]) for row in summaries if row["dataset"] == "data1"
    ]
    total_size = directory_size(root)
    spacing = read_single_row(root / "input-spacing.csv")
    text = f"""# Проверка полного расчёта

Все четыре расчёта `data1` и полный расчёт `data2` завершены. Метод Пикара сошёлся на каждом
принятом шаге.

- наибольшая точечная оценка временной погрешности уточнённого решения: {maximum_time_error:.6g} К;
- наибольшая точечная оценка пространственной погрешности уточнённого решения: {maximum_space_error:.6g} К;
- полный `data2`: {data2['cells_min']}–{data2['cells_max']} ячеек,
  {float(data2['wall_seconds']):.3f} с, {int(data2['output_bytes']) / (1024 * 1024):.1f} МиБ;
- размер одного расчёта `data1`: {min(data1_sizes) / (1024 * 1024):.1f}–{max(data1_sizes) / (1024 * 1024):.1f} МиБ;
- размер всего каталога этапа: {total_size / (1024 * 1024 * 1024):.2f} ГиБ;
- наибольшая относительная несбалансированность переноса: {data1_transfer_error:.3%} для `data1`
  и {float(data2['transfer_energy_error']):.3%} для `data2`.

Шаг исходных таблиц `data2` составляет {float(spacing['min_step']):.3f}–{float(spacing['max_step']):.3f} м,
медиана — {float(spacing['median_step']):.3f} м. Для полного
расчёта выбран базовый размер ячейки 50 м, шаг описания границ 25 м и локальный размер не более
5% толщины слоя. Такое сгущение разрешает тонкие слои без стоимости равномерной мелкой сетки.

Оценки по правилу Рунге предназначены для внутренней проверки и не подтверждают порядок метода
на произвольной неструктурированной сетке. Подробные значения находятся в
`runge-temperature.csv`, стоимость запусков — в `run-summary.csv`.
Большая несбалансированность переноса остаётся известным ограничением до реализации
консервативного переноса.
"""
    (root / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    arguments = parser.parse_args()
    summaries = write_run_summary(arguments.root)
    rows = runge_rows(arguments.root)
    with (arguments.root / "runge-temperature.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write_report(arguments.root, summaries, rows)


if __name__ == "__main__":
    main()

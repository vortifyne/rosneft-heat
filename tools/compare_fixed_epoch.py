#!/usr/bin/env python3
"""Compare two fixed-epoch solutions on their common physical sample lattice."""

import csv
import math
import pathlib
import sys


def read(path: pathlib.Path):
    values = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if not row["temperature"]:
                continue
            key = (float(row["x"]), float(row["z"]))
            material = int(row["material"]) if row["material"] else None
            values[key] = (float(row["temperature"]), material)
    return values


def metrics(differences):
    if not differences:
        return (0, math.nan, math.nan, math.nan, math.nan, math.nan)
    ordered = sorted(differences, key=lambda item: item[0])
    maximum, point = ordered[-1]
    rms = math.sqrt(sum(value * value for value, _ in ordered) / len(ordered))
    p95 = ordered[math.ceil(0.95 * len(ordered)) - 1][0]
    return (len(ordered), maximum, rms, p95, point[0], point[1])


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: compare_fixed_epoch.py LEFT_SAMPLES RIGHT_SAMPLES")
    left_path, right_path = map(pathlib.Path, sys.argv[1:])
    left, right = read(left_path), read(right_path)
    common = left.keys() & right.keys()
    all_differences = [(abs(left[key][0] - right[key][0]), key) for key in common]
    same_material = [
        (abs(left[key][0] - right[key][0]), key)
        for key in common
        if left[key][1] == right[key][1]
    ]
    print(
        "left,right,common,max_K,rms_K,p95_K,max_x,max_z,"
        "same_material,same_material_max_K,same_material_rms_K,"
        "same_material_p95_K,same_material_max_x,same_material_max_z"
    )
    print(
        ",".join(
            map(
                str,
                (
                    left_path.parent.name,
                    right_path.parent.name,
                    *metrics(all_differences),
                    *metrics(same_material),
                ),
            )
        )
    )


if __name__ == "__main__":
    main()

"""Sample geological layer geometry on a common horizontal grid."""

from __future__ import annotations

import numpy as np


def sampling_points(snapshot: list[dict[str, object]], step: float) -> np.ndarray:
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("Cell size must be finite and positive")
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

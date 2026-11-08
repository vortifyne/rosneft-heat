"""Read binary snapshots of the basin forward problem."""

from __future__ import annotations

from pathlib import Path
import struct

import numpy as np


def read_snapshot(path: Path) -> dict:
    with path.open("rb") as source:
        signature = source.read(8)
        if signature not in (b"CGRID001", b"CGRID002"):
            raise ValueError(f"Unsupported basin snapshot: {path}")
        # uint32 phase, float64 age (Ma), uint64 x/cell counts, uint32 reaction count.
        kind, age, x_count, cell_count, reaction_count = struct.unpack("<IdQQI", source.read(32))

        def read_array(dtype: str, count: int) -> np.ndarray:
            result = np.fromfile(source, dtype=dtype, count=count)
            if result.size != count:
                raise ValueError(f"Truncated basin snapshot: {path}")
            return result

        x = read_array("<f8", x_count)
        column = read_array("<u8", cell_count)
        layer = read_array("<i4", cell_count)
        top = read_array("<f8", cell_count)
        bottom = read_array("<f8", cell_count)
        temperature = read_array("<f8", cell_count)
        point_count = cell_count if signature == b"CGRID001" else cell_count + x_count - 1
        reflectance = read_array("<f8", point_count)
        integrals = read_array("<f8", point_count * reaction_count).reshape(point_count, reaction_count)
        if source.read(1):
            raise ValueError(f"Unexpected bytes in basin snapshot: {path}")
    return dict(path=path, kind=kind, age=age, x=x, column=column, layer=layer,
                top=top, bottom=bottom, temperature=temperature,
                reflectance=reflectance, integrals=integrals, nodal_maturity=signature == b"CGRID002")


def cell_reflectance(snapshot: dict) -> np.ndarray:
    """Cell averages for polygon display; saved nodal values remain available."""
    values = snapshot["reflectance"]
    if not snapshot["nodal_maturity"]:
        return values
    top_node = np.arange(len(snapshot["column"]), dtype=np.uint64) + snapshot["column"]
    return .5 * (values[top_node] + values[top_node + 1])

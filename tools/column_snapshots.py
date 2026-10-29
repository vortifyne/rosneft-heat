"""Read binary snapshots of the column-grid forward problem."""

from __future__ import annotations

from pathlib import Path
import struct

import numpy as np


def read_snapshot(path: Path) -> dict:
    with path.open("rb") as source:
        header = source.read(8)
        if header != b"CGRID001":
            raise ValueError(f"Unsupported column-grid snapshot: {path}")
        kind, age, nx, cells, reactions = struct.unpack("<IdQQI", source.read(32))

        def array(dtype: str, count: int) -> np.ndarray:
            result = np.fromfile(source, dtype=dtype, count=count)
            if result.size != count:
                raise ValueError(f"Truncated column-grid snapshot: {path}")
            return result

        x = array("<f8", nx)
        column = array("<u8", cells)
        layer = array("<i4", cells)
        top = array("<f8", cells)
        bottom = array("<f8", cells)
        temperature = array("<f8", cells)
        reflectance = array("<f8", cells)
        integrals = array("<f8", cells * reactions).reshape(cells, reactions)
        if source.read(1):
            raise ValueError(f"Unexpected bytes in column-grid snapshot: {path}")
    return dict(path=path, kind=kind, age=age, x=x, column=column, layer=layer,
                top=top, bottom=bottom, temperature=temperature,
                reflectance=reflectance, integrals=integrals)

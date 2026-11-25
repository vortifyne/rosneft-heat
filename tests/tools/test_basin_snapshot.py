"""Snapshot maturity values map to the correct cells in all supported formats."""
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from basin_snapshot import cell_reflectance, read_snapshot


class SnapshotTest(unittest.TestCase):
    def test_cell_reflectance_across_layer_and_column_boundaries(self):
        layout = [([0., 1., 2.], "<f8"), ([0, 0, 0, 1], "<u8"),
                  ([1, 1, 2, 2], "<i4"), ([0., 1., 2., 0.], "<f8"),
                  ([1., 2., 3., 3.], "<f8"), ([300., 310., 320., 310.], "<f8")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.cgrd"
            for version, nodes, expected in [(1, 4, [0., 1., 2., 3.]),
                                             (2, 6, [.5, 1.5, 2.5, 4.5]),
                                             (3, 7, [.5, 1.5, 3.5, 5.5])]:
                with self.subTest(version=version):
                    reflectance = np.arange(nodes, dtype="<f8")
                    integrals = np.arange(2 * nodes, dtype="<f8").reshape(nodes, 2)
                    header = f"CGRID{version:03}".encode() + struct.pack("<IdQQI", 2, 0., 3, 4, 2)
                    cells = b"".join(np.array(data, dtype=dtype).tobytes() for data, dtype in layout)
                    path.write_bytes(header + cells + reflectance.tobytes() + integrals.tobytes())
                    snapshot = read_snapshot(path)
                    np.testing.assert_array_equal(cell_reflectance(snapshot), expected)
                    np.testing.assert_array_equal(snapshot["integrals"], integrals)

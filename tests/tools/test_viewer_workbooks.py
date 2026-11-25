from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'desktop'))
from viewer_data import load_directory


class ViewerWorkbookTest(unittest.TestCase):
    def test_dense_blocks_and_sparse_excel_fallback(self):
        columns = ['X', 'Time_Ma', 'Z_Top [m]', 'Z_Bot [m]', 'Z_Aver [m]',
                   'Top [C]', 'Bot [C]', 'Aver [C]']
        values = np.arange(24000, dtype=float).reshape(-1, 8) / 7
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'Temperature.xlsx'
            for sparse in (False, True):
                if sparse:
                    values[1010, 5] = np.nan
                pd.DataFrame(values, columns=columns).to_excel(path, index=False)
                expected = pd.read_excel(path)
                with patch('pandas.read_excel', wraps=pd.read_excel) as general_reader:
                    data, props, units = load_directory(temp)
                    self.assertEqual(general_reader.called, sparse)
                self.assertEqual(props, ['Temperature'])
                self.assertEqual(units, {'Temperature': 'C'})
                pd.testing.assert_frame_equal(data['Temperature']['Sheet1'], expected,
                                              check_dtype=False, check_exact=True)

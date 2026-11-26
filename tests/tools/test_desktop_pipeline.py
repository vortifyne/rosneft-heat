"""Desktop boundary checks with independent malformed-input cases."""
import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'desktop'))
from pipeline import validate_input, validate_options


class DesktopValidationTest(unittest.TestCase):
    def test_rejects_nonfinite_steps_and_negative_rdp(self):
        for key in ('dx', 'dz', 'dt-ma'):
            for value in ('0', '-1', 'nan', 'inf', 'broken'):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_options({key: value})
        for value in ('-1', 'nan', 'inf'):
            with self.assertRaises(ValueError):
                validate_options({'rdp-tol': value})
        self.assertEqual(validate_options({'rdp-tol': '0'}), {'rdp-tol': '0'})

    def test_rejects_locked_parameters_and_unknown_methods(self):
        for options in ({'scheme': 'nmpfa'}, {'linear-method': 'unknown'}, {'pc': 'none'}):
            with self.assertRaises(ValueError):
                validate_options(options)

    def test_solver_tolerances_and_iteration_limits(self):
        for key in ('linear-maxiter', 'nonlinear-maxiter'):
            for value in ('0','-1','1.5','nan','2147483648'):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_options({key:value})
            self.assertEqual(validate_options({key:'1200'}), {key:'1200'})
        for key in ('linear-rtol','nonlinear-rtol'):
            for value in ('-1','1','inf'):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_options({key:value})
        for key in ('linear-atol','nonlinear-atol','linear-rtol','nonlinear-rtol'):
            self.assertEqual(validate_options({key:'0'}), {key:'0'})

    def test_reports_missing_files_and_corrupt_workbook(self):
        from pipeline import REQUIRED
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, 'Missing input files:.*Depth.xlsx'):
                validate_input(root)
            for filename in REQUIRED:
                (root / filename).write_bytes(b'not an Excel file')
            with self.assertRaisesRegex(ValueError, 'Cannot read workbook Depth.xlsx'):
                validate_input(root)


if __name__ == '__main__':
    unittest.main()

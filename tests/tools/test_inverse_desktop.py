import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'desktop'))
from inverse_pipeline import validate_options


class InverseOptions(unittest.TestCase):
    def test_limits_are_positive_integers(self):
        for value in ('1.5', '0', '-2', 'nan', 'inf', '2147483648'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_options({'maxfev': value})

    def test_residual_zero_disables_tolerance(self):
        self.assertEqual(validate_options({'residual-tol': '0'}), {'residual-tol': '0'})

    def test_forward_parameters_are_accepted_and_validated(self):
        self.assertEqual(validate_options({'dx':'100','linear-maxiter':'1200'}),
                         {'dx':'100','linear-maxiter':'1200'})
        with self.assertRaises(ValueError):
            validate_options({'nonlinear-maxiter':'1.5'})

    def test_flux_matches_native_physical_bounds(self):
        for value in ('49.9', '100.1'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_options({'initial-q': value})
        for value in ('50', '100'):
            self.assertEqual(validate_options({'initial-q': value}), {'initial-q':value})

    def test_normalized_options(self):
        self.assertEqual(validate_options({'initial-q':'75', 'maxiter':'1', 'maxfev':'2'}),
                         {'initial-q':'75', 'maxiter':'1', 'maxfev':'2'})

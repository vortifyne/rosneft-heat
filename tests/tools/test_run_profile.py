import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'desktop'))
from run_profile import RunProfile, read_profile


class RunProfileTest(unittest.TestCase):
    def profile(self, directory, inverse=False):
        clock = [10, 12, 15, 25, 29, 30]
        with patch('run_profile.perf_counter', side_effect=clock):
            profile = RunProfile()
            for key in ('validate', 'prepare', 'solve', 'export', 'check'):
                profile.stage(key)
            profile.save(directory, inverse=inverse)
        return read_profile(directory)

    def test_forward_excludes_parents_and_nested_native_timers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'timings.txt').write_text(
                'Program 9.0 100\n  Forward solve 8.0 89\n    Nonlinear solve 7.0 78\n'
                '      Assembly 3.0 33\n        Matrix 2.0 22\n      Linear solve 4.0 44\n')
            (root / 'Temperature.xlsx').write_bytes(b'x' * 24)
            report = self.profile(root)
            self.assertEqual(report['counts']['exported_workbooks'], 1)
            self.assertEqual(report['counts']['exported_bytes'], 24)
            self.assertEqual(report['stages_s'], dict(validate=2, prepare=3, solve=10, export=4, check=1))
            self.assertEqual(report['total_s'], 20)
            self.assertEqual(report['solver_s'], {
                'Assembly': 3, 'Linear solve': 4, 'Other solver and process time': 3})

    def test_inverse_uses_summary_not_final_forward_counters(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'summary.txt').write_text('elapsed_s=7.5\nforward_evaluations=12\nobservations=50\n')
            report = self.profile(root, inverse=True)
            self.assertEqual(report['counts']['forward_evaluations'], 12)
            self.assertEqual(report['counts']['observations'], 50)
            self.assertEqual(sorted(report['solver_s'].values()), [2.5, 7.5])

    def test_missing_native_timings_keeps_stage_report(self):
        with tempfile.TemporaryDirectory() as temp:
            report = self.profile(Path(temp))
            self.assertEqual(report['solver_s'], {})
            self.assertEqual(sum(report['stages_s'].values()), report['total_s'])

"""Regression tests for a silent or signal-resistant solver process."""
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'desktop'))
from pipeline import Cancelled, Pipeline


class ProcessSupervisionTest(unittest.TestCase):
    def test_both_solvers_only_emit_process_output_after_command(self):
        from inverse_pipeline import Pipeline as InversePipeline
        code = "import time; print('Iteration 0',flush=True); time.sleep(.2); print('Finished')"
        for engine in (Pipeline, InversePipeline):
            with self.subTest(engine=engine):
                lines = []
                # Cover a long quiet interval without slowing the suite down.
                with patch('pipeline.time', wraps=time) as clock:
                    clock.monotonic.side_effect = lambda: time.monotonic() * 200
                    engine()._execute([sys.executable, '-c', code], lines.append)
                self.assertEqual(lines[1:], ['Iteration 0', 'Finished'])

    def test_inverse_still_enforces_mpi_startup_timeout(self):
        from inverse_pipeline import Pipeline as InversePipeline
        lines = []
        with patch('pipeline._STOP_GRACE_S', .1):
            with self.assertRaisesRegex(RuntimeError, 'did not initialize'):
                InversePipeline()._execute([sys.executable, '-c', 'import time; time.sleep(60)'],
                                          lines.append, initialization_timeout=.15)
        self.assertTrue(any('initialization timed out' in line for line in lines))

    def test_preserves_partial_utf8_and_final_line(self):
        lines = []
        runner = Pipeline()
        code = "import os; os.write(1,b'first\\n\\xd0'); os.write(1,b'\\xb0\\nlast')"
        runner._execute([sys.executable, '-c', code], lines.append)
        self.assertEqual(lines[1:], ['first', 'а', 'last'])
        self.assertIsNone(runner.process)

    def check_cancel(self, fork=False):
        with tempfile.TemporaryDirectory() as temporary:
            ready = Path(temporary) / 'ready'
            code = ("import os,signal,sys,time; from pathlib import Path; "
                    "signal.signal(signal.SIGTERM, signal.SIG_IGN); " +
                    ("os.fork(); " if fork else "") +
                    "Path(sys.argv[1]).touch(); time.sleep(60)")
            runner, lines, errors = Pipeline(), [], []
            def work():
                try:
                    runner._execute([sys.executable, '-c', code, str(ready)], lines.append)
                except BaseException as error:
                    errors.append(error)
            with patch('pipeline._STOP_GRACE_S', 0.15):
                thread = threading.Thread(target=work, daemon=True)
                thread.start()
                try:
                    deadline = time.monotonic() + 3
                    while not ready.exists() and time.monotonic() < deadline:
                        time.sleep(0.01)
                    self.assertTrue(ready.exists(), errors)
                    started = time.monotonic()
                    runner.cancel()
                    thread.join(3)
                    self.assertFalse(thread.is_alive(), lines)
                    self.assertLess(time.monotonic() - started, 2)
                    self.assertEqual(len(errors), 1)
                    self.assertIsInstance(errors[0], Cancelled)
                    self.assertTrue(any('SIGKILL' in line for line in lines))
                    self.assertIsNone(runner.process)
                finally:
                    if runner.process is not None:
                        try:
                            os.killpg(runner.process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    thread.join(2)

    def test_cancels_silent_process_ignoring_sigterm(self):
        self.check_cancel()

    def test_cancels_group_with_inherited_stdout(self):
        self.check_cancel(fork=True)

    def test_times_out_mpi_startup_even_when_stdout_is_silent(self):
        lines = []
        runner = Pipeline()
        code = 'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)'
        with patch('pipeline._STOP_GRACE_S', 0.1):
            with self.assertRaisesRegex(RuntimeError, 'did not initialize'):
                runner._execute([sys.executable, '-c', code], lines.append,
                                initialization_timeout=0.2)
        self.assertTrue(any('initialization timed out' in line for line in lines))
        self.assertIsNone(runner.process)

    def test_startup_deadline_does_not_limit_computation(self):
        lines = []
        code = "import time; print('PETSc/MPI initialized',flush=True); time.sleep(0.4); print('done')"
        Pipeline()._execute([sys.executable, '-c', code], lines.append,
                            initialization_timeout=0.2)
        self.assertEqual(lines[-1], 'done')


if __name__ == '__main__':
    unittest.main()

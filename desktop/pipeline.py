"""Native XLSX -> existing C++ -> XLSX pipeline, shared by desktop and CLI."""
from __future__ import annotations

from datetime import datetime
import codecs
import json
import math
import os
from pathlib import Path
import signal
import selectors
import subprocess
import sys
import threading
import time
import uuid

EDITABLE = ('dx', 'dz', 'dt-ma', 'rdp-tol', 'linear-method', 'pc',
            'linear-rtol', 'linear-atol', 'linear-maxiter',
            'nonlinear-rtol', 'nonlinear-atol', 'nonlinear-maxiter')
CHOICES = {'linear-method': ('gmres', 'bicgstab', 'lu'),
           'pc': ('ilu', 'ilu2', 'gamg', 'boomeramg')}
REQUIRED = ('Depth.xlsx', 'Lithology.xlsx', 'Porosity.xlsx', 'Tsurf.xlsx',
            'Heat_Flow_Basal.xlsx', 'lithotype_properties.xlsx')
_STOP_GRACE_S = 3.0


def root_dir() -> Path:
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).resolve().parents[2]
    return Path(__file__).resolve().parents[1]


def solver_command(inverse=False) -> list[str]:
    root = root_dir()
    launcher = root / ('run_inverse_solver' if inverse else 'run_solver')
    if getattr(sys, 'frozen', False):
        return [str(launcher)]
    return [str(root / 'build/docker/release' / ('basin_inverse_solver' if inverse else 'basin_forward_solver'))]


def self_command() -> list[str]:
    return [str(root_dir() / 'AppRun')] if getattr(sys, 'frozen', False) else [sys.executable, str(Path(__file__).with_name('main.py'))]


def child_environment() -> dict[str, str]:
    env = os.environ.copy()
    if getattr(sys, 'frozen', False):
        # The host shell starts before AppRun selects our bundled ELF loader.
        env.pop('LD_LIBRARY_PATH', None)
        env.pop('LD_PRELOAD', None)
    return env


def read_defaults(inverse=False) -> dict[str, str]:
    result = subprocess.run([*solver_command(inverse), '--print-defaults'], check=True,
                            capture_output=True, text=True, timeout=30, env=child_environment())
    return {k.replace('_', '-'): v for k, v in
            (line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)}


def validate_options(options: dict[str, str]) -> dict[str, str]:
    if set(options) - set(EDITABLE):
        raise ValueError('Unknown editable parameters: ' + ', '.join(sorted(set(options) - set(EDITABLE))))
    checked = {}
    for key, value in options.items():
        value = str(value).strip()
        if key in CHOICES:
            if value not in CHOICES[key]:
                raise ValueError(f'{key}: select one of {", ".join(CHOICES[key])}')
        else:
            try:
                number = float(value)
            except ValueError:
                raise ValueError(f'{key}: enter a number') from None
            nonnegative = key == 'rdp-tol' or key.endswith(('-rtol', '-atol'))
            if not math.isfinite(number) or (number < 0 if nonnegative else number <= 0):
                raise ValueError(f'{key}: expected a finite {"non-negative" if nonnegative else "positive"} value')
            if key.endswith('-rtol') and number >= 1:
                raise ValueError(f'{key}: relative tolerance must be less than 1')
            if key.endswith('-maxiter'):
                if not number.is_integer() or number > 2147483647:
                    raise ValueError(f'{key}: expected a positive integer up to 2147483647')
                value = str(int(number))
            else:
                value = format(number, '.17g')
        checked[key] = value
    return checked


def create_run_directory(source: Path, output: Path, run_name: str | None = None) -> Path:
    if output == source or source in output.parents:
        raise ValueError('Choose an output directory outside the input directory')
    if output == root_dir() or root_dir() in output.parents:
        raise ValueError('Choose an output directory outside the application installation')
    if run_name is None:
        run_name = datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
    if not run_name.strip() or run_name in ('.', '..') or Path(run_name).name != run_name:
        raise ValueError('Enter a single result folder name.')
    output.mkdir(parents=True, exist_ok=True)
    directory = output / f'{source.name}_run-{run_name}'
    try:
        directory.mkdir()
    except FileExistsError:
        raise ValueError(f'Result folder already exists: {directory}. Choose another name.') from None
    return directory


def validate_input(directory: Path, inverse=False) -> None:
    """Validate workbook structures before invoking the existing converter."""
    import numpy as np
    import pandas as pd
    if not directory.is_dir():
        raise ValueError(f'Input directory does not exist: {directory}')
    required = tuple(name for name in REQUIRED if name != 'Heat_Flow_Basal.xlsx') if inverse else REQUIRED
    if inverse:
        required += ('TemperatureWellData.xlsx', 'VitriniteWellData.xlsx')
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise ValueError('Missing input files: ' + ', '.join(missing))

    def workbook(filename, **kwargs):
        try:
            return pd.read_excel(directory / filename, **kwargs)
        except Exception as error:
            raise ValueError(f'Cannot read workbook {filename}: {error}') from error

    geometry = workbook('Depth.xlsx', sheet_name=None)
    if not geometry:
        raise ValueError('Depth.xlsx contains no layers')
    for name, frame in geometry.items():
        if frame.empty or frame.shape[1] < 5:
            raise ValueError(f'Depth.xlsx / {name}: expected geometry columns')
        values = frame.iloc[:, :5].to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any(values[:, 3] < values[:, 2]):
            raise ValueError(f'Depth.xlsx / {name}: invalid depths or missing values')
        for age, rows in frame.groupby(frame.columns[1], sort=False):
            if len(rows) < 2 or not np.all(np.diff(rows.iloc[:, 0].to_numpy(float)) > 0):
                raise ValueError(f'Depth.xlsx / {name} / {age}: X must increase strictly')
    ages = {float(age) for frame in geometry.values() for age in frame.iloc[:, 1]}
    if len(ages) < 2:
        raise ValueError('At least two configuration ages are required')
    lithology_codes = set()
    for filename in (name for name in REQUIRED[1:-1]
                     if not inverse or name != 'Heat_Flow_Basal.xlsx' or (directory / name).is_file()):
        pages = workbook(filename, sheet_name=None)
        if set(pages) != set(geometry):
            raise ValueError(f'{filename}: layer sheets do not match Depth.xlsx')
        for name, frame in pages.items():
            if frame.shape[1] != 8 or len(frame) != len(geometry[name]):
                raise ValueError(f'{filename} / {name}: expected 8 columns and matching rows')
            values = frame.to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f'{filename} / {name}: non-numeric or missing values')
            if not np.array_equal(values[:, :5], geometry[name].iloc[:, :5].to_numpy(float)):
                raise ValueError(f'{filename} / {name}: geometry or row order differs from Depth.xlsx')
            if filename == 'Porosity.xlsx' and ((values[:, 5:] < 0).any() or (values[:, 5:] > 100).any()):
                raise ValueError(f'{filename} / {name}: porosity must lie within 0..100 percent')
            if filename == 'Lithology.xlsx':
                lithology_codes.update(values[:, 7].astype(int))
    properties = workbook(REQUIRED[-1])
    if properties.empty or properties.shape[1] < 10:
        raise ValueError('lithotype_properties.xlsx: expected 10 property columns')
    if not np.isfinite(properties.iloc[:, [0, 2, 3, 5, 9]].to_numpy(float)).all():
        raise ValueError('lithotype_properties.xlsx: material properties must be finite')
    codes = set(properties.iloc[:, 0].astype(int))
    # The organizer's reader deliberately casts the Aver column to int.
    # Real workbooks contain half-integer averages; preserve that convention.
    if not lithology_codes <= codes:
        raise ValueError('Lithology.xlsx contains unknown lithotype codes')
    if inverse:
        import re
        count = 0
        for filename in ('TemperatureWellData.xlsx', 'VitriniteWellData.xlsx'):
            frame = workbook(filename)
            if frame.empty or len(frame.columns) % 2:
                raise ValueError(f'{filename}: expected pairs of depth/value columns')
            for column in range(0, len(frame.columns), 2):
                if not re.search(r'\(([0-9.]+) m\)', str(frame.columns[column])):
                    raise ValueError(f'{filename}: missing well X coordinate in column title')
                pair = frame.iloc[:, column:column + 2]
                if (pair.isna().any(axis=1) & ~pair.isna().all(axis=1)).any():
                    raise ValueError(f'{filename}: incomplete depth/value observation')
                values = pair.dropna().to_numpy(dtype=float)
                if not np.isfinite(values).all() or (values[:, 1] == 0).any():
                    raise ValueError(f'{filename}: observations must be finite with nonzero measurement error')
                count += len(values)
        if not count:
            raise ValueError('No well observations found')



class Cancelled(Exception):
    pass


class Pipeline:
    def __init__(self, emit=lambda line: None):
        self.emit = emit
        self.cancelled = threading.Event()
        self.process = None
        self.lock = threading.Lock()
        self.directory = None

    def cancel(self):
        self.cancelled.set()
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    def _execute(self, command, write, initialization_timeout=None):
        """Supervise a process without blocking on a silent stdout pipe."""
        if self.cancelled.is_set():
            raise Cancelled()
        write('Command: ' + json.dumps(command, ensure_ascii=False))
        env = child_environment()
        env.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', BLIS_NUM_THREADS='1')
        with self.lock:
            if self.cancelled.is_set():
                raise Cancelled()
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, env=env, start_new_session=True, bufsize=0)
            process = self.process

        def send(signal_number):
            try:
                os.killpg(process.pid, signal_number)
            except ProcessLookupError:
                pass

        pending = ''
        initialized = initialization_timeout is None
        decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')

        def consume(data, final=False):
            nonlocal pending, initialized
            pending += decoder.decode(data, final=final)
            while '\n' in pending:
                line, pending = pending.split('\n', 1)
                if line.strip() == 'PETSc/MPI initialized':
                    initialized = True
                write(line.rstrip())
            if final and pending:
                write(pending.rstrip())
                pending = ''

        started = time.monotonic()
        stopped = forced = exited = None
        timed_out = eof = False
        try:
            os.set_blocking(process.stdout.fileno(), False)
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while True:
                    now = time.monotonic()
                    code = process.poll()
                    if (not initialized and not timed_out and not self.cancelled.is_set()
                            and code is None and now - started >= initialization_timeout):
                        timed_out = True
                        write('ERROR: PETSc/MPI initialization timed out; stopping solver')
                    if self.cancelled.is_set() or timed_out:
                        if stopped is None:
                            stopped = now
                            write(f'Stopping process group {process.pid} (SIGTERM)')
                            send(signal.SIGTERM)
                        if forced is None and now - stopped >= _STOP_GRACE_S:
                            forced = now
                            write(f'Force-stopping process group {process.pid} (SIGKILL)')
                            send(signal.SIGKILL)
                        if forced is not None and code is None and now - forced >= 2:
                            raise RuntimeError(f'Cannot stop process {process.pid} after SIGKILL')
                    if code is not None:
                        if exited is None:
                            exited = now
                        if eof:
                            break
                        # A descendant may retain stdout after the parent exits.
                        # The reader must never keep the GUI in a running state forever.
                        if now - exited >= 1:
                            send(signal.SIGKILL)
                            write('Warning: child output pipe remained open after process exit; closed')
                            break
                    for key, _ in selector.select(timeout=0.1):
                        try:
                            data = os.read(key.fd, 65536)
                        except BlockingIOError:
                            continue
                        if data:
                            consume(data)
                        else:
                            selector.unregister(key.fileobj)
                            eof = True
            consume(b'', final=True)
            code = process.wait(timeout=2)
            if self.cancelled.is_set():
                raise Cancelled()
            if timed_out:
                raise RuntimeError('PETSc/MPI did not initialize within the startup time limit. '
                                   'No calculation results were produced; see run.log.')
            if code:
                raise RuntimeError(f'Process failed with exit code {code}')
        finally:
            if process.poll() is None:
                send(signal.SIGKILL)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    write(f'ERROR: process {process.pid} did not exit after SIGKILL')
            process.stdout.close()
            with self.lock:
                self.process = None

    def run(self, source: Path, output: Path, options: dict[str, str], run_name: str | None = None) -> Path:
        source, output = source.resolve(), output.resolve()
        options = validate_options(options)
        if not source.is_dir():
            raise ValueError(f'Input directory does not exist: {source}')
        directory = create_run_directory(source, output, run_name)
        self.directory = directory
        with (directory / 'run.log').open('w', encoding='utf-8', buffering=1) as log:
            def write(message):
                line = f'[{datetime.now().astimezone().isoformat(timespec="seconds")}] {message}'
                log.write(line + '\n')
                self.emit(line)

            def execute(command, **kwargs):
                return self._execute(command, write, **kwargs)

            status = 'error'
            try:
                write('STAGE: Validating Excel workbooks')
                execute([*self_command(), '--validate', str(source)])
                write('STAGE: Preparing input data')
                execute([*self_command(), '--prepare', str(source), str(directory / 'input')])
                defaults = read_defaults()
                settings = dict(defaults, **options)
                (directory / 'configuration.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
                arguments = []
                for key, value in settings.items():
                    if value != 'none':
                        arguments.extend(['--' + key, value])
                write('STAGE: Calculating temperature and EASY%Ro')
                execute([*solver_command(), str(directory / 'input'), str(directory), *arguments],
                        initialization_timeout=60)
                write('STAGE: Exporting Excel results')
                execute([*self_command(), '--export', str(source), str(directory)])
                write('STAGE: Checking results')
                execute([*self_command(), '--check-results', str(source), str(directory)])
                status = 'success'
                write('SUCCESS: Calculation completed')
                return directory
            except Cancelled:
                status = 'cancelled'
                write('CANCELLED: Calculation stopped; partial files are diagnostic only')
                raise
            except Exception as error:
                write(f'ERROR: {error}')
                raise
            finally:
                (directory / 'status.json').write_text(json.dumps({
                    'status': status, 'finished': datetime.now().astimezone().isoformat(),
                    'input': str(source), 'output': str(directory)}, indent=2), encoding='utf-8')


def check_results(source: Path, directory: Path, *, inverse=False):
    """Require complete XLSX outputs matching the native fields and recovered flux."""
    import numpy as np
    import pandas as pd
    from basin_data import _layer_sort_key
    from basin_snapshot import read_snapshot
    geometry = pd.read_excel(source / 'Depth.xlsx', sheet_name=None)
    field_directory = directory / 'forward' if inverse else directory
    fields = pd.read_csv(field_directory / 'fields.csv', float_precision='round_trip')
    fields = fields.set_index(['age_ma', 'layer_id', 'x'])
    layer_ids = {name: index + 1 for index, name in enumerate(sorted(geometry, key=_layer_sort_key))}
    outputs = [('Temperature.xlsx', 'temperature'),
               ('Sweeney_Burnham_1990_EASY_Ro.xlsx', 'reflectance')]
    if inverse:
        flux = pd.read_csv(directory / 'parameters.csv', float_precision='round_trip').set_index('age_ma')
        outputs.append(('Heat_Flow_Basal.xlsx', 'flux'))
    for name, quantity in outputs:
        frames = pd.read_excel(directory / name, sheet_name=None)
        if set(frames) != set(geometry):
            raise ValueError(f'{name}: missing output layers')
        for layer, frame in frames.items():
            if frame.shape[1] != 8 or not np.array_equal(frame.iloc[:, :5], geometry[layer].iloc[:, :5]):
                raise ValueError(f'{name} / {layer}: output geometry does not match input')
            actual = frame.iloc[:, 5:].to_numpy(float)
            if not np.isfinite(actual).all():
                raise ValueError(f'{name} / {layer}: invalid calculated values')
            if quantity == 'flux':
                q = frame.iloc[:, 1].map(flux.basal_heat_flux_mw_m2).to_numpy()
                expected = np.column_stack((q, q, q))
            else:
                index = pd.MultiIndex.from_arrays([
                    frame.iloc[:, 1], np.full(len(frame), layer_ids[layer]), frame.iloc[:, 0]])
                columns = [f'{quantity}_{part}' for part in ('top', 'bottom', 'average')]
                expected = fields.loc[index, columns].to_numpy()
            if not np.allclose(actual, expected, rtol=1e-12, atol=1e-10):
                raise ValueError(f'{name} / {layer}: Excel values differ from native solver output')
    paths = list((field_directory / 'snapshots').glob('snapshot_*.cgrd'))
    if not paths:
        raise ValueError('No calculation snapshots were produced')
    for path in paths:
        snapshot = read_snapshot(path)
        if not np.isfinite(snapshot['temperature']).all() or not np.isfinite(snapshot['reflectance']).all():
            raise ValueError(f'Invalid snapshot values: {path.name}')
    print(f'Verified {len(outputs)} Excel files, {len(geometry)} layers and {len(paths)} snapshots', flush=True)

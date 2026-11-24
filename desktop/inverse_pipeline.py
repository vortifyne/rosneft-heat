"""Desktop orchestration for the existing basin inverse solver and its plots."""
from datetime import datetime
import json
import math
from pathlib import Path
import uuid

from pipeline import Pipeline as ForwardPipeline, Cancelled, root_dir, self_command, solver_command
from pipeline import read_defaults as native_defaults, validate_input as common_validate_input
from pipeline import check_results as check_field_results
from pipeline import EDITABLE as FORWARD_PARAMETERS, CHOICES, validate_options as validate_forward_options

INVERSE_PARAMETERS = ('initial-q', 'maxiter', 'maxfev', 'residual-tol')
EDITABLE = (*FORWARD_PARAMETERS, *INVERSE_PARAMETERS)


def read_defaults():
    return native_defaults(inverse=True)


def validate_input(directory):
    common_validate_input(directory, inverse=True)


def validate_options(options):
    if set(options) - set(EDITABLE):
        raise ValueError('Неизвестные параметры: ' + ', '.join(sorted(set(options) - set(EDITABLE))))
    checked = validate_forward_options({key: value for key, value in options.items() if key in FORWARD_PARAMETERS})
    for key, raw in options.items():
        if key in FORWARD_PARAMETERS:
            continue
        try:
            value = float(str(raw).strip())
        except ValueError:
            raise ValueError(f'{key}: введите число') from None
        if not math.isfinite(value) or (value < 0 if key == 'residual-tol' else value <= 0):
            raise ValueError(f'{key}: недопустимое значение')
        if key == 'initial-q' and not 50 <= value <= 100:
            raise ValueError('initial-q: ожидается от 50 до 100 мВт/м² (границы решателя).')
        if key in ('maxiter', 'maxfev'):
            if not value.is_integer() or value > 2147483647:
                raise ValueError(f'{key}: введите положительное целое число до 2147483647')
            checked[key] = str(int(value))
        else:
            checked[key] = format(value, '.17g')
    return checked

def check_results(source, directory):
    import numpy as np
    import pandas as pd
    tables = {name: pd.read_csv(directory / name) for name in
              ('parameters.csv', 'initial_parameters.csv', 'observations.csv', 'progress.csv')}
    for name, frame in tables.items():
        numbers = frame.select_dtypes(include='number')
        if frame.empty or numbers.empty or not np.isfinite(numbers.to_numpy()).all():
            raise ValueError(f'{name}: отсутствуют или некорректны результаты')
    original = pd.read_csv(directory / 'input/observations.csv')
    compared = tables['observations.csv']
    if len(original) != len(compared):
        raise ValueError('В результатах потеряны скважинные наблюдения')
    for column in ('kind', 'well', 'x', 'z'):
        matches = (np.allclose(original[column], compared[column], rtol=0, atol=1e-8) if column in ('x', 'z')
                   else np.array_equal(original[column].to_numpy(), compared[column].to_numpy()))
        if not matches:
            raise ValueError(f'Наблюдения не совпадают: {column}')
    if not np.allclose(compared.calculated - compared.observed,
                       compared.error * compared.normalized_residual, atol=1e-10):
        raise ValueError('Некорректная нормированная невязка')
    configurations = pd.read_csv(directory / 'input/configurations.csv')
    if len(tables['parameters.csv']) != len(configurations):
        raise ValueError('Количество восстановленных параметров не совпадает с эпохами')
    summary = dict(line.split('=', 1) for line in (directory / 'summary.txt').read_text().splitlines() if '=' in line)
    if summary['termination'] not in ('measurement_tolerance', 'optimizer_converged', 'iteration_limit', 'evaluation_limit'):
        raise ValueError('Оптимизатор завершился с ошибкой: ' + summary['termination'])
    check_field_results(source, directory, inverse=True)
    print(f'Verified {len(configurations)} parameters and {len(compared)} observations; '
          f'termination={summary["termination"]}', flush=True)


class Pipeline(ForwardPipeline):
    report_idle = False

    def run(self, source, output, options):
        source, output = Path(source).resolve(), Path(output).resolve()
        options = validate_options(options)
        if not source.is_dir():
            raise ValueError(f'Input directory does not exist: {source}')
        if output == source or source in output.parents:
            raise ValueError('Choose an output directory outside the input directory')
        if output == root_dir() or root_dir() in output.parents:
            raise ValueError('Choose an output directory outside the application installation')
        output.mkdir(parents=True, exist_ok=True)
        directory = output / ('run-' + datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6])
        directory.mkdir()
        self.directory = directory
        with (directory / 'run.log').open('w', encoding='utf-8', buffering=1) as log:
            def write(message):
                line = f'[{datetime.now().astimezone().isoformat(timespec="seconds")}] {message}'
                log.write(line + '\n')
                self.emit(line)

            status = 'error'
            try:
                write('STAGE: Validating Excel workbooks')
                self._execute([*self_command(), '--inverse', '--validate', str(source)], write)
                write('STAGE: Preparing input data')
                self._execute([*self_command(), '--inverse', '--prepare', str(source), str(directory / 'input')], write)
                settings = dict(read_defaults(), **options)
                # The inverse GUI shows the final fields for the recovered q(t).
                # Native fields.csv still covers every original configuration age.
                settings['snapshots'] = 'final'
                (directory / 'configuration.json').write_text(json.dumps(settings, indent=2))
                arguments = [item for key, value in settings.items() if value != 'none'
                             for item in ('--' + key, value)]
                write('STAGE: Recovering basal heat flux')
                self._execute([*solver_command(inverse=True), str(directory / 'input'), str(directory), '--save-forward', *arguments],
                              write, initialization_timeout=60)
                write('STAGE: Exporting Excel results')
                self._execute([*self_command(), '--inverse', '--export', str(source), str(directory)], write)
                write('STAGE: Checking results')
                self._execute([*self_command(), '--inverse', '--check-results', str(source), str(directory)], write)
                status = 'success'
                write('SUCCESS: Inverse calculation completed; see summary.txt for termination reason')
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
                    'input': str(source), 'output': str(directory)}, indent=2))

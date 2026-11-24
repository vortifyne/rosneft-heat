"""Small wall-clock report written once per successful desktop calculation."""
import json
import math
from time import perf_counter


STAGES = {
    'validate': ('Validate workbooks', 'Open the input workbooks and check their structure and values.'),
    'prepare': ('Prepare solver input', 'Read the Excel data and convert it to the native solver format.'),
    'solve': ('Run solver', 'Start the native solver, perform the calculation and write its outputs.'),
    'export': ('Export Excel results', 'Map the calculated fields onto the original layers and write the result workbooks.'),
    'check': ('Verify results', 'Read the exported workbooks back, compare them with native fields and check snapshots.'),
}
NATIVE_ROWS = {
    'Initialization and shutdown', 'Input', 'Initial stationary solve',
    'Geometry, grid and step preparation', 'State transfer', 'Assembly',
    'Linear solve', 'EASY%Ro', 'Output',
}


class RunProfile:
    def __init__(self):
        self.boundaries = []

    def stage(self, key):
        self.boundaries.append((key, perf_counter()))

    def save(self, directory, inverse=False):
        end = perf_counter()
        stages = {}
        stop = end
        for key, start in reversed(self.boundaries):
            stages[key] = stop - start
            stop = start
        report = dict(version=1, mode='inverse' if inverse else 'forward',
                      total_s=end - self.boundaries[0][1], stages_s=stages,
                      solver_s={}, counts={})
        workbooks = list(directory.glob('*.xlsx'))
        report['counts']['exported_workbooks'] = len(workbooks)
        report['counts']['exported_bytes'] = sum(path.stat().st_size for path in workbooks)
        summary = directory / 'summary.txt'
        if summary.is_file():
            fields = dict(line.split('=', 1) for line in summary.read_text().splitlines() if '=' in line)
            for key in ('timesteps', 'linear_iterations', 'nonlinear_iterations', 'forward_evaluations',
                        'observations', 'parameters', 'iterations'):
                value = fields.get(key, '')
                if value.isdecimal():
                    report['counts'][key] = int(value)
            if inverse:
                record_time(report, 'Input, inversion and parameter output', fields.get('elapsed_s'))
        timing = directory / 'timings.txt'
        if not inverse and timing.is_file():
            with timing.open() as stream:
                for line in stream:
                    if line.startswith('Properties, total'):
                        break
                    row = line.rsplit(None, 2)
                    if len(row) == 3 and row[0].strip() in NATIVE_ROWS:
                        record_time(report, row[0].strip(), row[1])
        detail = report['solver_s']
        if detail:
            remaining = stages['solve'] - sum(detail.values())
            if remaining >= -0.00001:
                label = 'Final forward run, startup and other work' if inverse else 'Other solver and process time'
                detail[label] = max(0, remaining)
            else:
                detail.clear()  # Incompatible native measurements cannot form a breakdown
        temp = directory / 'performance.json.tmp'
        temp.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        temp.replace(directory / 'performance.json')


def record_time(report, label, value):
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return
    if math.isfinite(seconds) and seconds >= 0:
        report['solver_s'][label] = seconds


def read_profile(directory):
    report = json.loads((directory / 'performance.json').read_text(encoding='utf-8'))
    if not isinstance(report, dict) or report.get('version') != 1:
        raise ValueError('Unsupported performance report version.')
    for key in ('stages_s', 'solver_s', 'counts'):
        values = report.get(key)
        if not isinstance(values, dict) or not all(type(v) in (int, float) and math.isfinite(v) and v >= 0
                                                    for v in values.values()):
            raise ValueError('Invalid performance measurements.')
    total = report.get('total_s')
    if (type(total) not in (int, float) or not math.isfinite(total) or total <= 0
            or not STAGES.keys() <= report['stages_s'].keys()):
        raise ValueError('Incomplete performance report.')
    return report

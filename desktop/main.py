"""Single Basin Solver entrypoint: Qt UI or isolated computation helpers."""
from __future__ import annotations
import os
from pathlib import Path
import sys
import tempfile

if not getattr(sys, 'frozen', False):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
else:
    os.environ.pop('LD_LIBRARY_PATH', None)
    os.environ.pop('LD_PRELOAD', None)
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / f'basin-mpl-{os.getuid()}'))
os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'


def dispatch():
    arguments = sys.argv[1:]
    inverse = '--inverse' in arguments
    if inverse:
        arguments.remove('--inverse')
    if not arguments or arguments[0] not in ('--validate','--prepare','--export','--check-results','--defaults','--run'):
        return False
    import pipeline
    if inverse:
        import inverse_pipeline as engine
    else:
        engine = pipeline
    action, *args = arguments
    if action == '--validate':
        engine.validate_input(Path(args[0])); print('Input workbooks validated', flush=True)
    elif action == '--prepare':
        import prepare_basin_input
        sys.argv = ['prepare_basin_input', *args]
        from viewer_data import read_layers

        def load_property(path):
            print('Читаю ' + path)
            return read_layers(path)

        prepare_basin_input.main(load_property)
    elif action == '--export':
        from export_basin_results import export_results
        export_results(Path(args[0]), Path(args[1]), inverse=inverse)
    elif action == '--check-results':
        engine.check_results(Path(args[0]), Path(args[1]))
    elif action == '--defaults':
        import json
        print(json.dumps(engine.read_defaults(), indent=2))
    else:
        import argparse, signal
        parser = argparse.ArgumentParser()
        parser.add_argument('input', type=Path); parser.add_argument('output', type=Path)
        for key in engine.EDITABLE:
            parser.add_argument('--' + key)
        options = parser.parse_args(args)
        values = {key:getattr(options,key.replace('-','_')) for key in engine.EDITABLE
                  if getattr(options,key.replace('-','_')) is not None}
        runner = engine.Pipeline(lambda line: print(line, flush=True))
        signal.signal(signal.SIGTERM, lambda *_: runner.cancel())
        signal.signal(signal.SIGINT, lambda *_: runner.cancel())
        print(runner.run(options.input, options.output, values), flush=True)
    return True


if __name__ == '__main__':
    try:
        if not dispatch():
            from ui import run_gui
            run_gui()
    except Exception as error:
        print(f'ERROR: {error}', file=sys.stderr, flush=True)
        sys.exit(1)

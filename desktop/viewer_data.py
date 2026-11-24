"""Read organizer viewer workbooks without writing to or unpickling input folders."""
from pathlib import Path


def load_directory(directory, emit=lambda _: None, cancelled=lambda: False):
    import pandas as pd
    def _sheet_matches_schema(frame):
        if not {'X','Time_Ma','Z_Top [m]','Z_Bot [m]','Z_Aver [m]'}.issubset(frame.columns):
            return False
        return all(any(str(column).startswith(part) and '[' in str(column)
                       for column in frame.columns) for part in ('Top','Bot','Aver'))
    data, properties, units = {}, [], {}
    files = sorted(path for path in Path(directory).iterdir()
                   if path.suffix.lower() == '.xlsx' and not path.name.startswith('~$'))
    if not files:
        raise ValueError('The selected folder contains no XLSX files.')
    for path in files:
        if cancelled():
            raise InterruptedError('Loading stopped')
        emit(f'Loading {path.name}')
        try:
            with pd.ExcelFile(path) as book:
                layers = {}
                for sheet in book.sheet_names:
                    if cancelled():
                        raise InterruptedError('Loading stopped')
                    frame = pd.read_excel(book, sheet)
                    if not _sheet_matches_schema(frame):
                        layers = {}
                        break
                    layers[sheet] = frame
            if not layers:
                emit(f'Skipped {path.name}: not a layer-property workbook')
                continue
            prop = path.stem
            properties.append(prop)
            data[prop] = layers
            column = next(name for name in next(iter(layers.values())).columns
                          if name.startswith('Aver') and '[' in name)
            units[prop] = column.split('[', 1)[1].rstrip(']')
        except InterruptedError:
            raise
        except Exception as error:
            raise ValueError(f'{path.name}: {error}') from error
    if not properties:
        raise ValueError('No compatible layer-property workbooks were found.')
    return data, properties, units

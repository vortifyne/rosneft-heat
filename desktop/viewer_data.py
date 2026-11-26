"""Read organizer viewer workbooks without writing to the input folder."""
from pathlib import Path
import re

_NUMBER = rb'([-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[Ee][-+]?[0-9]+)?)'
_ROW = re.compile(rb'<row r="([0-9]+)">' + b''.join(
    rb'<c r="' + bytes([letter]) + rb'\1" t="n"><v>' + _NUMBER + rb'</v></c>'
    for letter in range(ord('A'), ord('I'))) + rb'</row>')


def numeric_sheet(book, sheet, columns, cancelled):
    """Decode blocks of canonical numeric XML; return None on another layout."""
    import numpy as np
    import pandas as pd
    if len(columns) != 8 or not hasattr(sheet, '_worksheet_path'):
        return None
    with book._archive.open(sheet._worksheet_path) as raw:
        prefix = raw.read(65536)
        _, marker, rest = prefix.partition(b'<sheetData><row r="1">')
        if not marker or b'</row>' not in rest:
            return None
        rest = rest.split(b'</row>', 1)[1]
        chunks = []
        next_row = 2
        while True:
            if cancelled():
                raise InterruptedError('Loading stopped')
            body, end, _ = rest.partition(b'</sheetData>')
            stop = len(body) if end else body.rfind(b'</row>') + len(b'</row>')
            if not end and stop < len(b'</row>'):
                stop = 0
            pos = 0
            values = bytearray()
            for match in _ROW.finditer(body, 0, stop):
                if match.start() != pos or int(match[1]) != next_row:
                    return None
                next_row += 1
                values.extend(b' '.join(match.groups()[1:]))
                values.append(32)
                pos = match.end()
            if pos != stop:
                return None
            if values:
                chunks.append(np.fromstring(bytes(values), sep=' ').reshape(-1, 8))
            if end:
                break
            block = raw.read(512 * 1024)
            if not block or len(body) - stop > 65536:
                return None
            rest = body[stop:] + block
        if chunks:
            return pd.DataFrame(np.concatenate(chunks), columns=columns)
        return None


def compatible(columns):
    required = {'X', 'Time_Ma', 'Z_Top [m]', 'Z_Bot [m]', 'Z_Aver [m]'}
    return required.issubset(columns) and all(
        any(str(c).startswith(part) and '[' in str(c) for c in columns)
        for part in ('Top', 'Bot', 'Aver'))


def read_layers(path, *, properties_only=False, cancelled=lambda: False):
    """Read dense organizer sheets directly and delegate other layouts to pandas."""
    import pandas as pd
    with pd.ExcelFile(path, engine='openpyxl') as book:
        layers = {}
        for sheet in book.book.worksheets:
            if cancelled():
                raise InterruptedError('Loading stopped')
            sheet.reset_dimensions()
            columns = next(sheet.values, ())
            is_property = compatible(columns)
            if properties_only and not is_property:
                return {}
            frame = None
            if (is_property and all(isinstance(c, str) and c for c in columns)
                    and len(set(columns)) == len(columns)):
                frame = numeric_sheet(book.book, sheet, columns, cancelled)
            if frame is None:
                frame = pd.read_excel(book, sheet.title)
            layers[sheet.title] = frame
        return layers


def load_directory(directory, emit=lambda _: None, cancelled=lambda: False):
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
            layers = read_layers(path, properties_only=True, cancelled=cancelled)
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

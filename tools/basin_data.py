"""
Читалка данных для задачи Академического турнира 2026:
двумерная модель теплопереноса в осадочном чехле.

Загружает 6 xlsx-файлов и предоставляет удобный API для доступа
к геометрии, литологии, пористости, граничным условиям и свойствам литотипов.

Использование:
    from basin_data import BasinData

    data = BasinData("C:/path/to/xlsx_dir")

    # Временные шаги и слои
    data.times           # [0.0, 5.0, 10.0, ..., 200.0]
    data.layers          # ['Layer_1', 'Layer_2', ..., 'Layer_7']

    # Геометрия слоя в момент времени
    x, z_top, z_bot = data.geometry('Layer_5', time=75.0)

    # Литотип, пористость
    lith_codes = data.lithology('Layer_5', time=75.0)   # массив кодов
    porosity   = data.porosity('Layer_5', time=75.0)     # массив φ (в %)

    # Граничные условия
    x, tsurf   = data.tsurf(time=0.0)         # Tsurf(x) [°C]
    x, qbasal  = data.heat_flow(time=0.0)     # qbasal(x) [mW/m²]

    # Свойства литотипов
    data.lithotypes
    # {1001: {'name': 'Halite_new', 'rho_s': 2200.0, 'lambda_20': 6.5,
    #         'c20_J': 862.5, 'As': 0.0124, ...}, ...}

    # Какие слои существуют в данный момент
    data.layers_at(time=75.0)  # ['Layer_5', 'Layer_5_1', ..., 'Layer_7']

    # Кинетика EASY%Ro (Sweeney & Burnham, 1990)
    data.easy_ro  # {'A': [...], 'E': [...], 'f': [...]}
"""

import os
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------
# Кинетические константы EASY%Ro (Sweeney & Burnham, 1990)
# -----------------------------------------------------------------------

_EASY_RO = {
    # Энергии активации Ei [ккал/моль]
    'E_kcal': np.array([
        34, 36, 38, 40, 42, 44, 46, 48, 50, 52,
        54, 56, 58, 60, 62, 64, 66, 68, 70, 72,
    ], dtype=np.float64),

    # Весовые доли fi
    'f': np.array([
        0.03, 0.03, 0.04, 0.04, 0.05, 0.05, 0.06, 0.04, 0.04, 0.07,
        0.06, 0.06, 0.06, 0.05, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01,
    ], dtype=np.float64),

    # Предэкспоненциальный множитель A [1/с] — единый для всех реакций
    'A': 1.0e13,

    # Универсальная газовая постоянная Rg [ккал/(моль·К)]
    'Rg_kcal': 1.987e-3,
}

# Перевод в СИ
_EASY_RO['E_J'] = _EASY_RO['E_kcal'] * 4186.8     # Дж/моль
_EASY_RO['Rg_J'] = 8.314                            # Дж/(моль·К)


# -----------------------------------------------------------------------
# Вспомогательные функции
# -----------------------------------------------------------------------

def _layer_sort_key(name):
    """Ключ сортировки: 'Layer_4_1' -> (4, 1)."""
    parts = name.replace('Layer_', '').split('_')
    return tuple(int(p) for p in parts)


def _load_property_xlsx(filepath):
    """
    Загружает xlsx со свойством.
    Возвращает dict: layer_name -> DataFrame.
    """
    print("Читаю " + filepath)
    xls = pd.ExcelFile(filepath)
    result = {}
    for sheet in xls.sheet_names:
        result[sheet] = pd.read_excel(xls, sheet)
    return result


# -----------------------------------------------------------------------
# Основной класс
# -----------------------------------------------------------------------

class BasinData:
    """Контейнер данных бассейна для задачи теплопереноса."""

    def __init__(self, xlsx_dir):
        """
        Параметры:
            xlsx_dir       : папка с Depth.xlsx, Lithology.xlsx, Porosity.xlsx,
                             Heat_Flow_Basal.xlsx, Tsurf.xlsx
        """
        # Загрузка свойств-полей
        self._depth = _load_property_xlsx(os.path.join(xlsx_dir, 'Depth.xlsx'))
        self._lith = _load_property_xlsx(os.path.join(xlsx_dir, 'Lithology.xlsx'))
        self._poro = _load_property_xlsx(os.path.join(xlsx_dir, 'Porosity.xlsx'))
        self._hf = _load_property_xlsx(os.path.join(xlsx_dir, 'Heat_Flow_Basal.xlsx'))
        self._tsurf = _load_property_xlsx(os.path.join(xlsx_dir, 'Tsurf.xlsx'))

        # Слои (общий список из Depth)
        self.layers = sorted(self._depth.keys(), key=_layer_sort_key)

        # Временные шаги (объединение всех)
        all_times = set()
        self._layer_times = {}
        for layer, df in self._depth.items():
            ts = sorted(df.iloc[:, 1].unique())
            self._layer_times[layer] = ts
            all_times.update(ts)
        self.times = sorted(all_times)

        # Свойства литотипов
        self.lithotypes = self._load_lithotypes(os.path.join(xlsx_dir, 'lithotype_properties.xlsx'))

        # Кинетика EASY%Ro
        self.easy_ro = _EASY_RO.copy()

    # -------------------------------------------------------------------
    # Загрузка литотипов
    # -------------------------------------------------------------------

    @staticmethod
    def _load_lithotypes(filepath):
        """Загружает lithotype_properties.xlsx в словарь {ID: {...}}."""
        df = pd.read_excel(filepath)
        result = {}
        for _, row in df.iterrows():
            lid = int(row.iloc[0])
            result[lid] = {
                'name':      row.iloc[1],
                'rho_s':     float(row.iloc[2]),     # кг/м³
                'lambda_20': float(row.iloc[3]),     # Вт/(м·К)
                'c20_kcal':  float(row.iloc[4]),     # ккал/(кг·К)
                'c20_J':     float(row.iloc[5]),     # Дж/(кг·К)
                'U_ppm':     float(row.iloc[6]),
                'Th_ppm':    float(row.iloc[7]),
                'K_pct':     float(row.iloc[8]),
                'As':        float(row.iloc[9]),     # µW/m³
            }
        return result

    # -------------------------------------------------------------------
    # Доступ к данным
    # -------------------------------------------------------------------

    def layers_at(self, time):
        """Список слоёв, существующих в момент time (отсортировано сверху вниз)."""
        return [l for l in self.layers if time in self._layer_times.get(l, [])]

    def _get_slice(self, storage, layer, time):
        """Извлекает строки для (layer, time) из хранилища."""
        df = storage.get(layer)
        if df is None:
            raise KeyError(f'Слой {layer} не найден')
        sub = df[df.iloc[:, 1] == time]
        if len(sub) == 0:
            raise KeyError(
                f'Слой {layer} не существует в момент {time} Ma'
            )
        return sub

    def geometry(self, layer, time):
        """
        Геометрия слоя.

        Возвращает:
            x     : np.array — координата вдоль разреза [м]
            z_top : np.array — глубина кровли [м]
            z_bot : np.array — глубина подошвы [м]
        """
        sub = self._get_slice(self._depth, layer, time)
        return (sub.iloc[:, 0].values.astype(np.float64),
                sub.iloc[:, 2].values.astype(np.float64),
                sub.iloc[:, 3].values.astype(np.float64))

    def lithology(self, layer, time):
        """Коды литотипов (Aver). Возвращает np.array of int."""
        sub = self._get_slice(self._lith, layer, time)
        return sub.iloc[:, 7].values.astype(int)

    def porosity(self, layer, time):
        """
        Пористость (Aver).

        Возвращает:
            φ : np.array — пористость в **процентах**
                Для формул (2)–(4) делите на 100.
        """
        sub = self._get_slice(self._poro, layer, time)
        return sub.iloc[:, 7].values.astype(np.float64)

    def tsurf(self, time):
        """
        Температура поверхности Tsurf(x, t) — граничное условие (8).

        Возвращает:
            x     : np.array [м]
            tsurf : np.array [°C]

        Берётся из самого верхнего слоя в момент time.
        """
        top_layer = self.layers_at(time)[0]
        sub = self._get_slice(self._tsurf, top_layer, time)
        return (sub.iloc[:, 0].values.astype(np.float64),
                sub.iloc[:, 7].values.astype(np.float64))

    def heat_flow(self, time):
        """
        Базальный тепловой поток qbasal(x, t) — граничное условие (9).

        Возвращает:
            x      : np.array [м]
            qbasal : np.array [мВт/м²]

        Берётся из самого нижнего слоя в момент time.
        """
        bot_layer = self.layers_at(time)[-1]
        sub = self._get_slice(self._hf, bot_layer, time)
        return (sub.iloc[:, 0].values.astype(np.float64),
                sub.iloc[:, 7].values.astype(np.float64))

    # -------------------------------------------------------------------
    # Полная конфигурация на момент времени
    # -------------------------------------------------------------------

    def snapshot(self, time):
        """
        Полный снимок бассейна в момент time.

        Возвращает list of dict (сверху вниз), каждый элемент:
            {
                'layer':    str,
                'x':        np.array,
                'z_top':    np.array,
                'z_bot':    np.array,
                'lith':     np.array (int),
                'porosity': np.array (%, делить на 100 для формул),
            }
        """
        result = []
        for layer in self.layers_at(time):
            x, zt, zb = self.geometry(layer, time)
            result.append({
                'layer': layer,
                'x': x,
                'z_top': zt,
                'z_bot': zb,
                'lith': self.lithology(layer, time),
                'porosity': self.porosity(layer, time),
            })
        return result

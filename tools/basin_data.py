"""
Data reader for the Academic Tournament 2026:
two-dimensional heat transfer in a sedimentary basin.

Loads six XLSX files with geometry, lithology, porosity,
boundary conditions and lithotype properties.

Usage:
    from basin_data import BasinData

    data = BasinData("C:/path/to/xlsx_dir")

    # Ages and layers
    data.times           # [0.0, 5.0, 10.0, ..., 200.0]
    data.layers          # ['Layer_1', 'Layer_2', ..., 'Layer_7']

    # Layer geometry at a given age
    x, z_top, z_bot = data.geometry('Layer_5', time=75.0)

    # Lithotype codes and porosity
    lith_codes = data.lithology('Layer_5', time=75.0)   # code array
    porosity   = data.porosity('Layer_5', time=75.0)    # porosity [%]

    # Boundary conditions
    x, tsurf   = data.tsurf(time=0.0)         # Tsurf(x) [°C]
    x, qbasal  = data.heat_flow(time=0.0)     # qbasal(x) [mW/m²]

    # Lithotype properties
    data.lithotypes
    # {1001: {'name': 'Halite_new', 'rho_s': 2200.0, 'lambda_20': 6.5,
    #         'c20_J': 862.5, 'As': 0.0124, ...}, ...}

    # Layers present at a given age
    data.layers_at(time=75.0)  # ['Layer_5', 'Layer_5_1', ..., 'Layer_7']

    # EASY%Ro kinetics (Sweeney & Burnham, 1990)
    data.easy_ro  # {'A': 1e13, 'E_J': [...], 'f': [...]}
"""

import os
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------
# EASY%Ro kinetic constants (Sweeney & Burnham, 1990)
# -----------------------------------------------------------------------

_EASY_RO = {
    # Activation energies Ei [kcal/mol]
    'E_kcal': np.array([
        34, 36, 38, 40, 42, 44, 46, 48, 50, 52,
        54, 56, 58, 60, 62, 64, 66, 68, 70, 72,
    ], dtype=np.float64),

    # Reaction weights fi
    'f': np.array([
        0.03, 0.03, 0.04, 0.04, 0.05, 0.05, 0.06, 0.04, 0.04, 0.07,
        0.06, 0.06, 0.06, 0.05, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01,
    ], dtype=np.float64),

    # Preexponential factor A [1/s], shared by all reactions
    'A': 1.0e13,

    # Gas constant Rg [kcal/(mol K)]
    'Rg_kcal': 1.987e-3,
}

# Conversion to SI
_EASY_RO['E_J'] = _EASY_RO['E_kcal'] * 4186.8     # J/mol
_EASY_RO['Rg_J'] = 8.314                        # J/(mol K)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _layer_sort_key(name):
    """Sorting key: 'Layer_4_1' -> (4, 1)."""
    parts = name.replace('Layer_', '').split('_')
    return tuple(int(p) for p in parts)


def _load_property_xlsx(filepath):
    """
    Load an XLSX property file.
    Return a mapping from layer names to DataFrames.
    """
    print("Reading " + filepath)
    xls = pd.ExcelFile(filepath)
    result = {}
    for sheet in xls.sheet_names:
        result[sheet] = pd.read_excel(xls, sheet)
    return result


# -----------------------------------------------------------------------
# Basin data
# -----------------------------------------------------------------------

class BasinData:
    """Basin input data for the heat transfer problem."""

    def __init__(self, xlsx_dir):
        """
        Parameters:
            xlsx_dir       : directory containing Depth.xlsx, Lithology.xlsx, Porosity.xlsx,
                             Heat_Flow_Basal.xlsx, Tsurf.xlsx
        """
        # Load field data
        self._depth = _load_property_xlsx(os.path.join(xlsx_dir, 'Depth.xlsx'))
        self._lith = _load_property_xlsx(os.path.join(xlsx_dir, 'Lithology.xlsx'))
        self._poro = _load_property_xlsx(os.path.join(xlsx_dir, 'Porosity.xlsx'))
        self._hf = _load_property_xlsx(os.path.join(xlsx_dir, 'Heat_Flow_Basal.xlsx'))
        self._tsurf = _load_property_xlsx(os.path.join(xlsx_dir, 'Tsurf.xlsx'))

        # All layers from Depth
        self.layers = sorted(self._depth.keys(), key=_layer_sort_key)

        # Union of ages across all layers
        all_times = set()
        self._layer_times = {}
        for layer, df in self._depth.items():
            ts = sorted(df.iloc[:, 1].unique())
            self._layer_times[layer] = ts
            all_times.update(ts)
        self.times = sorted(all_times)

        # Lithotype properties
        self.lithotypes = self._load_lithotypes(os.path.join(xlsx_dir, 'lithotype_properties.xlsx'))

        # EASY%Ro kinetics
        self.easy_ro = _EASY_RO.copy()

    # -------------------------------------------------------------------
    # Lithotype loading
    # -------------------------------------------------------------------

    @staticmethod
    def _load_lithotypes(filepath):
        """Load lithotype_properties.xlsx into a mapping {ID: {...}}."""
        df = pd.read_excel(filepath)
        result = {}
        for _, row in df.iterrows():
            lid = int(row.iloc[0])
            result[lid] = {
                'name':      row.iloc[1],
                'rho_s':     float(row.iloc[2]),     # kg/m³
                'lambda_20': float(row.iloc[3]),     # W/(m K)
                'c20_kcal':  float(row.iloc[4]),     # kcal/(kg K)
                'c20_J':     float(row.iloc[5]),     # J/(kg K)
                'U_ppm':     float(row.iloc[6]),
                'Th_ppm':    float(row.iloc[7]),
                'K_pct':     float(row.iloc[8]),
                'As':        float(row.iloc[9]),     # µW/m³
            }
        return result

    # -------------------------------------------------------------------
    # Data access
    # -------------------------------------------------------------------

    def layers_at(self, time):
        """Return layers present at the given age, sorted from top to bottom."""
        return [l for l in self.layers if time in self._layer_times.get(l, [])]

    def _get_slice(self, storage, layer, time):
        """Return rows for the given layer and age."""
        df = storage.get(layer)
        if df is None:
            raise KeyError(f'Layer {layer} not found')
        sub = df[df.iloc[:, 1] == time]
        if len(sub) == 0:
            raise KeyError(
                f'Layer {layer} is not present at age {time} Ma'
            )
        return sub

    def geometry(self, layer, time):
        """
        Layer geometry.

        Returns:
            x     : np.array, horizontal coordinate [m]
            z_top : np.array, top depth [m]
            z_bot : np.array, bottom depth [m]
        """
        sub = self._get_slice(self._depth, layer, time)
        return (sub.iloc[:, 0].values.astype(np.float64),
                sub.iloc[:, 2].values.astype(np.float64),
                sub.iloc[:, 3].values.astype(np.float64))

    def lithology(self, layer, time):
        """Return lithotype codes (Aver) as an integer array."""
        sub = self._get_slice(self._lith, layer, time)
        return sub.iloc[:, 7].values.astype(int)

    def porosity(self, layer, time):
        """
        Porosity (Aver).

        Returns:
            phi : np.array, porosity in percent.
                  Divide by 100 for formulas (2)-(4).
        """
        sub = self._get_slice(self._poro, layer, time)
        return sub.iloc[:, 7].values.astype(np.float64)

    def tsurf(self, time):
        """
        Surface temperature Tsurf(x, t), boundary condition (8).

        Returns:
            x     : np.array [m]
            tsurf : np.array [°C]

        Read from the topmost layer at the given age.
        """
        top_layer = self.layers_at(time)[0]
        sub = self._get_slice(self._tsurf, top_layer, time)
        return (sub.iloc[:, 0].values.astype(np.float64),
                sub.iloc[:, 7].values.astype(np.float64))

    def heat_flow(self, time):
        """
        Basal heat flux qbasal(x, t), boundary condition (9).

        Returns:
            x      : np.array [m]
            qbasal : np.array [mW/m²]

        Read from the bottommost layer at the given age.
        """
        bot_layer = self.layers_at(time)[-1]
        sub = self._get_slice(self._hf, bot_layer, time)
        return (sub.iloc[:, 0].values.astype(np.float64),
                sub.iloc[:, 7].values.astype(np.float64))

    # -------------------------------------------------------------------
    # Full configuration at a given age
    # -------------------------------------------------------------------

    def snapshot(self, time):
        """
        Full basin configuration at the given age.

        Return a list of dictionaries ordered from top to bottom:
            {
                'layer':    str,
                'x':        np.array,
                'z_top':    np.array,
                'z_bot':    np.array,
                'lith':     np.array (int),
                'porosity': np.array (percent, divide by 100 for formulas),
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

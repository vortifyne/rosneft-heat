"""
Визуализатор 2D-разреза бассейна (Backstrip Boys team's optimization).

Зависимости: PyQt5, pyqtgraph, pandas, numpy, openpyxl
Backstrip Boys team note: matplotlib will add on the first run of Burial Plot instead of using everywhere.
"""

import sys
import os
import time
from collections import OrderedDict

import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from pyqtgraph.functions import arrayToQPath


# -----------------------------------------------------------------------
# Backstip boys optimization
# Constants
# -----------------------------------------------------------------------

N_SUB = 3                      # sub-bands per cell in Top->Bot mode
LUT_SIZE = 4096                # colour quantisation levels
CACHE_MAX_FRAMES = 24
CACHE_MAX_QUADS = 4_000_000    # rough memory cap for cached frames
PREFETCH_MAX_BUILD_S = 0.4     # skip neighbour prefetch if a build is slower

_CMAP_STOPS = [
    (0.0,   (  0,   0, 180)),
    (0.15,  (  0,  80, 220)),
    (0.30,  (  0, 180, 220)),
    (0.45,  (  0, 200, 150)),
    (0.55,  ( 50, 210,  50)),
    (0.70,  (150, 220,  30)),
    (0.85,  (220, 220,   0)),
    (1.0,   (255, 255,  50)),
]


# -----------------------------------------------------------------------
# Определение папки, где лежит exe / скрипт
# -----------------------------------------------------------------------

def get_app_dir():
    """
    Возвращает папку, в которой лежит exe (если приложение собрано
    PyInstaller-ом) или сам .py файл (при запуске обычным python).
    """
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


_REQUIRED_COLUMNS = {
    'X', 'Time_Ma', 'Z_Top [m]', 'Z_Bot [m]', 'Z_Aver [m]',
}


def _sheet_matches_schema(df):
    """Проверяет, что лист xlsx имеет структуру данных свойства слоя."""
    cols = set(df.columns)
    if not _REQUIRED_COLUMNS.issubset(cols):
        return False
    has_top = any(c.startswith('Top') and '[' in c for c in df.columns)
    has_bot = any(c.startswith('Bot') and '[' in c for c in df.columns)
    has_aver = any(c.startswith('Aver') and '[' in c for c in df.columns)
    return has_top and has_bot and has_aver


def layer_sort_key(name):
    parts = name.replace('Layer_', '').split('_')
    return tuple(int(p) for p in parts)


# -----------------------------------------------------------------------
# Backstip boys optimization
# Numeric core (no Qt): all per-frame work is vectorised over the whole
# frame instead of looping over layers / cells in Python
# -----------------------------------------------------------------------

def upper_bound(x, st, en, v):
    """For every segment [st, en) of ascending x: count of items <= v.
    Vectorised bisection over all segments at once."""
    lo = st.copy()
    hi = en.copy()
    if st.size == 0:
        return lo - st
    steps = int((en - st).max()).bit_length() + 1
    for _ in range(steps):
        act = lo < hi
        if not act.any():
            break
        mid = (lo + hi) >> 1
        right = act & (x[np.where(act, mid, 0)] <= v)
        lo = np.where(right, mid + 1, lo)
        hi = np.where(act & ~right, mid, hi)
    return lo - st


def nearest_rows(x, st, en, v):
    """Per segment: nearest node to v (ties go left), left node, next node,
    and whether v lies inside the segment's x range."""
    cnt = upper_bound(x, st, en, v)
    n = en - st
    left = np.clip(cnt - 1, 0, n - 1)
    nxt = np.minimum(left + 1, n - 1)
    xl = x[st + left]
    xr = x[st + nxt]
    near = np.where((left >= n - 1) | (v - xl <= xr - v), left, left + 1)
    inside = (v >= x[st]) & (v <= x[en - 1])
    return st + near, st + left, st + nxt, inside


class FrameView:
    """Segments (one per layer) of a single time step: views into PropData."""

    def __init__(self, pd, a, b):
        self.pd = pd
        self.st = pd.seg_start[a:b]
        self.en = pd.seg_end[a:b]
        self.layer = pd.seg_layer[a:b]
        self.r0 = int(self.st[0])
        self.r1 = int(self.en[-1])
        self.dict_rank = pd.dict_rank[self.layer]


class PropData:
    """All layers of one property packed into flat float64 arrays sorted by
    (time step, layer), so that any frame is a contiguous slice."""

    def __init__(self, layers):
        names = list(layers.keys())
        dict_pos = {n: i for i, n in enumerate(names)}
        self.names = sorted(names, key=layer_sort_key, reverse=True)
        self.dict_rank = np.array([dict_pos[n] for n in self.names],
                                  dtype=np.int64)

        chunks = []
        lens = []
        for n in self.names:
            a = layers[n].iloc[:, [0, 1, 2, 3, 5, 6, 7]].to_numpy(
                dtype=np.float64, na_value=np.nan)
            chunks.append(a)
            lens.append(a.shape[0])
        if chunks:
            allv = np.concatenate(chunks)
            lay = np.repeat(np.arange(len(chunks)), lens)
        else:
            allv = np.empty((0, 7))
            lay = np.empty(0, dtype=np.int64)

        ok = ~np.isnan(allv[:, 1])
        if not ok.all():
            allv = allv[ok]
            lay = lay[ok]

        ts = np.unique(allv[:, 1])
        T = ts.size
        self.n_frames = T
        self.times_desc = ts[::-1].copy()
        self.times = self.times_desc.tolist()
        self.frame_cache = {}

        if T == 0:
            self.x = self.zt = self.zb = self.vt = self.vb = self.va = \
                np.empty(0)
            self.seg_start = self.seg_end = self.seg_time = \
                self.seg_layer = np.empty(0, dtype=np.int64)
            self.pair_ok = np.empty(0, dtype=bool)
            self.ptr = np.zeros(1, dtype=np.int64)
            self.z_min, self.z_max = 0.0, 1.0
            self.v_min, self.v_max = 0.0, 1.0
            return

        frame = (T - 1) - np.searchsorted(ts, allv[:, 1])
        order = np.lexsort((lay, frame))
        allv = allv[order]
        lay = lay[order]
        frame = frame[order]
        R = allv.shape[0]

        col = np.ascontiguousarray
        self.x = col(allv[:, 0])
        self.zt = col(allv[:, 2])
        self.zb = col(allv[:, 3])
        self.vt = col(allv[:, 4])
        self.vb = col(allv[:, 5])
        self.va = col(allv[:, 6])

        chg = np.flatnonzero((frame[1:] != frame[:-1])
                             | (lay[1:] != lay[:-1])) + 1
        self.seg_start = np.concatenate(([0], chg)).astype(np.int64)
        self.seg_end = np.concatenate((chg, [R])).astype(np.int64)
        self.seg_time = frame[self.seg_start]
        self.seg_layer = lay[self.seg_start]
        self.ptr = np.searchsorted(self.seg_time, np.arange(T + 1))

        # pair_ok[i]: rows i and i+1 belong to the same layer and time step
        self.pair_ok = np.ones(max(R - 1, 0), dtype=bool)
        self.pair_ok[self.seg_end[:-1] - 1] = False

        z_min = min(np.nanmin(self.zt), np.nanmin(self.zb))
        z_max = max(np.nanmax(self.zt), np.nanmax(self.zb))
        margin = (z_max - z_min) * 0.03
        self.z_min = float(z_min - margin)
        self.z_max = float(z_max + margin)
        self.v_min = float(np.nanmin(self.va))
        self.v_max = float(np.nanmax(self.va))

    def frame(self, k):
        fr = self.frame_cache.get(k)
        if fr is None:
            fr = FrameView(self, self.ptr[k], self.ptr[k + 1])
            self.frame_cache[k] = fr
        return fr


def build_quads(pd, fr, mode, n_sub=N_SUB):
    """All cells of a frame as (M,4) x / z arrays plus one value per quad."""
    i0 = np.flatnonzero(pd.pair_ok[fr.r0:fr.r1 - 1]) + fr.r0
    if i0.size == 0:
        return None
    i1 = i0 + 1
    xl, xr = pd.x[i0], pd.x[i1]
    ztl, ztr = pd.zt[i0], pd.zt[i1]
    zbl, zbr = pd.zb[i0], pd.zb[i1]

    if mode == 'interp' and n_sub > 1:
        f = np.arange(n_sub + 1) / n_sub
        f0 = f[:-1][None, :]
        f1 = f[1:][None, :]
        fm = (f0 + f1) / 2.0
        dl = (zbl - ztl)[:, None]
        dr = (zbr - ztr)[:, None]
        z0l = ztl[:, None] + f0 * dl
        z1l = ztl[:, None] + f1 * dl
        z0r = ztr[:, None] + f0 * dr
        z1r = ztr[:, None] + f1 * dr
        vtl, vbl = pd.vt[i0][:, None], pd.vb[i0][:, None]
        vtr, vbr = pd.vt[i1][:, None], pd.vb[i1][:, None]
        val = (vtl + fm * (vbl - vtl) + vtr + fm * (vbr - vtr)) / 2.0
        pz = np.stack([z0l, z0r, z1r, z1l], axis=-1).reshape(-1, 4)
        px = np.repeat(np.stack([xl, xr, xr, xl], axis=-1), n_sub, axis=0)
        val = val.reshape(-1)
    else:
        px = np.stack([xl, xr, xr, xl], axis=-1)
        pz = np.stack([ztl, ztr, zbr, zbl], axis=-1)
        val = pd.va[i0]
    return px, pz, val


def burial_field(pd, sx, n_z=300):
    """Burial plot data at X = sx, vectorised over all layers and times."""
    if pd.seg_start.size == 0:
        return None
    rows, _, _, inside = nearest_rows(pd.x, pd.seg_start, pd.seg_end, sx)
    sel = np.flatnonzero(inside)
    if sel.size == 0:
        return None
    r = rows[sel]
    seg_t = pd.times_desc[pd.seg_time[sel]]
    seg_l = pd.seg_layer[sel]
    zt, zb, vt, vb = pd.zt[r], pd.zb[r], pd.vt[r], pd.vb[r]

    z_min_g = float(min(np.nanmin(zt), np.nanmin(zb)))
    z_max_g = float(max(np.nanmax(zt), np.nanmax(zb)))
    depth_grid = np.linspace(z_min_g, z_max_g, n_z)

    ages = pd.times_desc[::-1]
    age_fine = np.arange(ages.min(), ages.max() + 0.5, 1.0)
    age_fine = np.unique(np.concatenate([age_fine, ages]))
    field = np.full((age_fine.size, n_z), np.nan)

    order = np.lexsort((seg_t, seg_l))
    cuts = np.flatnonzero(np.diff(seg_l[order])) + 1
    dg = depth_grid[None, :]
    lines = []

    for g in np.split(order, cuts):
        t = seg_t[g]
        ztp, zbp, vtp, vbp = zt[g], zb[g], vt[g], vb[g]

        in_rng = (age_fine >= t[0]) & (age_fine <= t[-1])
        rows_f = np.flatnonzero(in_rng)
        ti = age_fine[in_rng]
        z0 = np.interp(ti, t, ztp)
        z1 = np.interp(ti, t, zbp)
        v0 = np.interp(ti, t, vtp)
        v1 = np.interp(ti, t, vbp)

        ins = (dg >= z0[:, None]) & (dg <= z1[:, None])
        th = (z1 - z0)[:, None]
        with np.errstate(divide='ignore', invalid='ignore'):
            vals = v0[:, None] + (dg - z0[:, None]) / th * (v1 - v0)[:, None]
        vals = np.where(th > 0, vals, v0[:, None])
        blk = field[rows_f]
        blk[ins] = vals[ins]
        field[rows_f] = blk

        tf = np.linspace(t[0], t[-1], 200)
        lines.append(np.column_stack([tf, np.interp(tf, t, ztp)]))
        lines.append(np.column_stack([tf, np.interp(tf, t, zbp)]))

    return age_fine, depth_grid, field, lines, z_min_g, z_max_g


def _import_mpl():
    try:
        import matplotlib
        matplotlib.use('Qt5Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.collections import LineCollection
    except ImportError:
        return None
    return plt, LinearSegmentedColormap, LineCollection


# -----------------------------------------------------------------------
# Графические элементы
# -----------------------------------------------------------------------

class FrameMeshItem(pg.GraphicsObject):
    """Whole section of one time step: QPainterPaths grouped by colour.
    Paths are kept as-is (no QPicture serialisation)."""

    def __init__(self, parts, brect):
        super().__init__()
        self._parts = parts
        self._brect = brect

    def paint(self, painter, *args):
        painter.save()
        painter.setRenderHint(QtGui.QPainter.Antialiasing, False)
        for path, pen, brush in self._parts:
            painter.setPen(pen)
            painter.setBrush(brush)
            painter.drawPath(path)
        painter.restore()

    def boundingRect(self):
        return self._brect


class HLinesItem(pg.GraphicsObject):
    """Many horizontal lines spanning the visible area, in one item."""

    def __init__(self, pen):
        super().__init__()
        self._pen = pen
        self._zs = np.empty(0)

    def set_zs(self, zs):
        self._zs = zs
        self.update()

    def boundingRect(self):
        r = self.viewRect()
        return QtCore.QRectF() if r is None else r

    def viewTransformChanged(self):
        super().viewTransformChanged()
        self.prepareGeometryChange()

    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        return (None, None)

    def paint(self, painter, *args):
        r = self.viewRect()
        if r is None or self._zs.size == 0:
            return
        painter.setPen(self._pen)
        x0, x1 = r.left(), r.right()
        for z in self._zs.tolist():
            painter.drawLine(QtCore.QPointF(x0, z), QtCore.QPointF(x1, z))


class ColorBarWidget(QtWidgets.QWidget):
    """Вертикальная цветовая шкала с подписями min/max."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(96)
        self.cmap = None
        self.vmin = 0.0
        self.vmax = 1.0
        self.unit = ''
        self._bar = None
        self._bar_key = None

    def set_range(self, vmin, vmax, unit, cmap):
        self.vmin = vmin
        self.vmax = vmax
        self.unit = unit
        self.cmap = cmap
        self.update()

    def _bar_pixmap(self, bar_w, bar_h):
        key = (id(self.cmap), bar_w, bar_h)
        if self._bar_key != key:
            t = 1.0 - np.arange(bar_h) / bar_h
            rgba = np.asarray(self.cmap.map(t, mode='byte'))
            pm = QtGui.QPixmap(bar_w + 1, bar_h)
            p = QtGui.QPainter(pm)
            for iy in range(bar_h):
                c = rgba[iy]
                p.setPen(QtGui.QColor(int(c[0]), int(c[1]),
                                      int(c[2]), int(c[3])))
                p.drawLine(0, iy, bar_w, iy)
            p.end()
            self._bar = pm
            self._bar_key = key
        return self._bar

    def paintEvent(self, event):
        if self.cmap is None:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        h = self.height()
        bar_x = 10
        bar_w = 25
        bar_top = 30
        bar_bot = h - 30
        bar_h = bar_bot - bar_top
        if bar_h <= 0:
            painter.end()
            return

        painter.drawPixmap(bar_x, bar_top, self._bar_pixmap(bar_w, bar_h))

        painter.setPen(QtGui.QColor(180, 180, 180))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawRect(bar_x, bar_top, bar_w, bar_h)

        painter.setPen(QtGui.QColor(220, 220, 220))
        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)

        txt_x = bar_x + bar_w + 5
        painter.drawText(txt_x, bar_top + 12, f'{self.vmax:.4g}')
        painter.drawText(txt_x, bar_bot, f'{self.vmin:.4g}')

        n_ticks = 4
        for i in range(1, n_ticks):
            frac = i / n_ticks
            yy = bar_top + int(frac * bar_h)
            val = self.vmax - frac * (self.vmax - self.vmin)
            painter.drawLine(bar_x + bar_w - 3, yy, bar_x + bar_w, yy)
            painter.drawText(txt_x, yy + 4, f'{val:.4g}')

        if self.unit:
            painter.drawText(bar_x, bar_top - 8, self.unit)

        painter.end()


class FlowLayout(QtWidgets.QLayout):
    """Let the original viewer controls wrap inside the portrait viewer screen."""
    def __init__(self):
        super().__init__()
        self.items = []
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(6)

    def addItem(self, item):
        self.items.append(item)

    def addSpacing(self, width):
        self.addItem(QtWidgets.QSpacerItem(width, 1))

    def addStretch(self, stretch=0):
        pass

    def count(self):
        return len(self.items)

    def itemAt(self, index):
        return self.items[index] if 0 <= index < len(self.items) else None

    def takeAt(self, index):
        return self.items.pop(index) if 0 <= index < len(self.items) else None

    def expandingDirections(self):
        return QtCore.Qt.Orientations(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self.arrange(QtCore.QRect(0, 0, width, 0), True)

    def minimumSize(self):
        size = QtCore.QSize()
        for item in self.items:
            size = size.expandedTo(item.minimumSize())
        return size

    def sizeHint(self):
        return self.minimumSize()

    def setGeometry(self, rectangle):
        super().setGeometry(rectangle)
        self.arrange(rectangle, False)

    def arrange(self, rectangle, measure):
        x, y, height = rectangle.x(), rectangle.y(), 0
        for item in self.items:
            size = item.sizeHint()
            if x + size.width() > rectangle.right() and height:
                x = rectangle.x(); y += height + self.spacing(); height = 0
            if not measure:
                item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), size))
            x += size.width() + self.spacing()
            height = max(height, size.height())
        return y + height - rectangle.y()


def _fix_y_range(widget, lo, hi):
    """setYRange only if the view is not already there."""
    vb = widget.getViewBox()
    cur = widget.viewRange()[1]
    tol = 1e-9 * max(1.0, abs(lo), abs(hi))
    if (vb.autoRangeEnabled()[1] or abs(cur[0] - lo) > tol
            or abs(cur[1] - hi) > tol):
        widget.setYRange(lo, hi, padding=0)


# -----------------------------------------------------------------------
# Главное окно
# -----------------------------------------------------------------------

class BasinViewer(QtWidgets.QWidget):

    def __init__(self, directory=None, loaded=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('2D Basin Viewer')
        self.resize(1200, 700)

        self.xlsx_dir = str(directory) if directory is not None else get_app_dir()
        self.data = {}          # prop_name -> {layer_name -> DataFrame}
        self.properties = []
        self.units = {}
        self.time_index = 0
        self.current_times = []

        self._pdata = {}                # prop -> PropData
        self._frame = None              # FrameView on screen (hover/profile)
        self._shown = []                # items of the current frame
        self._cache = OrderedDict()     # (prop, k, mode) -> (items, cost)
        self._cache_cost = 0
        self._last_build_s = 0.0
        self._draw_token = 0
        self._current_unit = ''
        self._prof_unit = None
        self._cross_visible = False
        self._last_hover_text = None

        self._cmap = pg.ColorMap([c[0] for c in _CMAP_STOPS],
                                 [c[1] for c in _CMAP_STOPS])
        self._build_color_lut()

        if loaded is None:
            self._load_data()
        else:
            self.data, self.properties, self.units = loaded
        self._init_ui()

        if self.properties:
            self._on_property_changed()

    # -------------------------------------------------------------------
    # Загрузка данных
    # -------------------------------------------------------------------

    def _load_data(self):
        from viewer_data import load_directory
        self.data, self.properties, self.units = load_directory(self.xlsx_dir)

    def _warn_no_valid_files(self, all_xlsx):
        QtWidgets.QMessageBox.critical(
            self, 'Ошибка',
            f'В папке {self.xlsx_dir} есть xlsx-файлы ({len(all_xlsx)} шт.), '
            f'но ни один не подходит по структуре для визуализации.\n\n'
            f'Нужны файлы со столбцами вида '
            f'X, Time_Ma, Z_Top, Z_Bot, Z_Aver, Top/Bot/Aver '
            f'(как Porosity.xlsx, Lithology.xlsx и т.п.).'
        )

    def _get_pdata(self, prop):
        pd_ = self._pdata.get(prop)
        if pd_ is None:
            pd_ = PropData(self.data[prop])
            self._pdata[prop] = pd_
        return pd_

    # -------------------------------------------------------------------
    # UI
    # -------------------------------------------------------------------

    def _init_ui(self):
        central = self

        toolbar = FlowLayout()

        toolbar.addWidget(QtWidgets.QLabel('Свойство:'))
        self.prop_combo = QtWidgets.QComboBox()
        self.prop_combo.addItems(self.properties)
        self.prop_combo.setMinimumWidth(130)
        self.prop_combo.currentIndexChanged.connect(self._on_property_changed)
        toolbar.addWidget(self.prop_combo)

        toolbar.addSpacing(10)

        self.btn_prev = QtWidgets.QPushButton('◀')
        self.btn_prev.setFixedWidth(40)
        self.btn_prev.clicked.connect(self._prev_time)
        toolbar.addWidget(self.btn_prev)

        self.time_label = QtWidgets.QLabel('0 Ma')
        self.time_label.setAlignment(QtCore.Qt.AlignCenter)
        self.time_label.setMinimumWidth(70)
        font = self.time_label.font()
        font.setPointSize(12)
        font.setBold(True)
        self.time_label.setFont(font)
        toolbar.addWidget(self.time_label)

        self.btn_next = QtWidgets.QPushButton('▶')
        self.btn_next.setFixedWidth(40)
        self.btn_next.clicked.connect(self._next_time)
        toolbar.addWidget(self.btn_next)

        toolbar.addStretch()

        toolbar.addWidget(QtWidgets.QLabel('Профиль:'))
        self.profile_mode = QtWidgets.QComboBox()
        self.profile_mode.addItems(['Aver', 'Top→Bot'])
        self.profile_mode.setFixedWidth(100)
        self.profile_mode.currentIndexChanged.connect(self._on_display_mode_changed)
        toolbar.addWidget(self.profile_mode)

        toolbar.addSpacing(8)

        self.btn_burial = QtWidgets.QPushButton('Burial Plot')
        self.btn_burial.clicked.connect(self._show_burial_plot)
        toolbar.addWidget(self.btn_burial)

        # ----- Plot -----
        pg.setConfigOptions(
            background=QtGui.QColor(30, 30, 30),
            foreground='w',
            antialias=True,
        )

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('bottom', 'X, m')
        self.plot_widget.setLabel('left', 'Depth, m')
        self.plot_widget.invertY(True)
        self.plot_widget.setAspectLocked(False)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        for axis_name in ('bottom', 'left'):
            self.plot_widget.getAxis(axis_name).enableAutoSIPrefix(False)

        # Crosshair and slice line stay in the scene; z-values keep them
        # above the (re-added) section items
        self._vline = pg.InfiniteLine(angle=90, pen=pg.mkPen('y', width=1,
                                      style=QtCore.Qt.DashLine))
        self._hline = pg.InfiniteLine(angle=0, pen=pg.mkPen('y', width=1,
                                      style=QtCore.Qt.DashLine))
        self._vline.setVisible(False)
        self._hline.setVisible(False)
        self._vline.setZValue(10)
        self._hline.setZValue(10)
        self.plot_widget.addItem(self._vline, ignoreBounds=True)
        self.plot_widget.addItem(self._hline, ignoreBounds=True)

        self.plot_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)

        self._slice_line = pg.InfiniteLine(
            angle=90, movable=True,
            pen=pg.mkPen(color=(255, 80, 80), width=2),
            hoverPen=pg.mkPen(color=(255, 120, 120), width=3),
        )
        self._slice_line.setZValue(11)
        self._slice_line.sigPositionChanged.connect(self._update_profile)
        self.plot_widget.addItem(self._slice_line, ignoreBounds=True)

        # ----- 1D профиль -----
        self.profile_widget = pg.PlotWidget()
        self.profile_widget.setLabel('bottom', 'Value')
        self.profile_widget.setLabel('left', 'Depth, m')
        self.profile_widget.invertY(True)
        self.profile_widget.showGrid(x=True, y=True, alpha=0.2)
        self.profile_widget.setMinimumWidth(160)
        self.profile_widget.setMaximumWidth(240)
        for axis_name in ('bottom', 'left'):
            self.profile_widget.getAxis(axis_name).enableAutoSIPrefix(False)

        # Persistent items: updated with setData instead of clear()+plot()
        self._prof_curve = pg.PlotCurveItem(
            pen=pg.mkPen(color=(100, 200, 255), width=2))
        self._prof_lines = HLinesItem(pg.mkPen((80, 80, 80), width=0.5))
        self.profile_widget.addItem(self._prof_curve)
        self.profile_widget.addItem(self._prof_lines, ignoreBounds=True)

        self.info_label = QtWidgets.QLabel('')
        self.info_label.setStyleSheet(
            'color: #cccccc; font-family: Consolas, monospace; font-size: 11px;'
            'padding: 3px 6px;'
        )

        self.colorbar = ColorBarWidget()

        plot_row = QtWidgets.QHBoxLayout()
        plot_row.addWidget(self.plot_widget, 1)
        plot_row.addWidget(self.colorbar)
        plot_row.addWidget(self.profile_widget)

        layout = QtWidgets.QVBoxLayout(central)
        layout.addLayout(toolbar)
        layout.addLayout(plot_row)
        layout.addWidget(self.info_label)

    # -------------------------------------------------------------------
    # Навигация по времени
    # -------------------------------------------------------------------

    def _gather_times(self, prop_name):
        """Уникальные временные шаги, от старого к молодому."""
        return self._get_pdata(prop_name).times

    def _on_property_changed(self):
        prop = self.properties[self.prop_combo.currentIndex()]

        old_time = (self.current_times[self.time_index]
                    if self.current_times else None)

        self.current_times = self._gather_times(prop)
        self._compute_fixed_ranges(prop)

        if old_time is not None and self.current_times:
            diffs = [abs(t - old_time) for t in self.current_times]
            self.time_index = diffs.index(min(diffs))
        else:
            self.time_index = 0

        self._update_buttons()
        self._draw()

    def _prev_time(self):
        if self.time_index > 0:
            self.time_index -= 1
            self._update_buttons()
            self._draw()

    def _next_time(self):
        if self.time_index < len(self.current_times) - 1:
            self.time_index += 1
            self._update_buttons()
            self._draw()

    def _update_buttons(self):
        self.btn_prev.setEnabled(self.time_index > 0)
        self.btn_next.setEnabled(
            self.time_index < len(self.current_times) - 1
        )
        if self.current_times:
            t = self.current_times[self.time_index]
            self.time_label.setText(f'{t:g} Ma')

    def _compute_fixed_ranges(self, prop_name):
        """Глобальные диапазоны глубин и значений по ВСЕМ временным шагам."""
        pd_ = self._get_pdata(prop_name)
        self._fixed_z_min = pd_.z_min
        self._fixed_z_max = pd_.z_max
        self._fixed_v_min = pd_.v_min
        self._fixed_v_max = pd_.v_max

    def _on_display_mode_changed(self):
        self._draw()

    # -------------------------------------------------------------------
    # Палитра
    # -------------------------------------------------------------------

    def _get_colormap(self):
        return self._cmap

    def _build_color_lut(self):
        """Quantised colour table: LUT index -> compact id of a distinct
        colour (a few hundred at most), plus the RGBA of every id."""
        t = np.linspace(0.0, 1.0, LUT_SIZE)
        rgba = np.asarray(self._cmap.map(t, mode='byte'), dtype=np.uint32)
        if rgba.shape[1] == 3:
            rgba = np.concatenate(
                [rgba, np.full((rgba.shape[0], 1), 255, np.uint32)], axis=1)
        packed = ((rgba[:, 0] << 24) | (rgba[:, 1] << 16)
                  | (rgba[:, 2] << 8) | rgba[:, 3])
        uniq, inv = np.unique(packed, return_inverse=True)
        self._lut_cid = inv.reshape(-1).astype(np.uint16)
        self._lut_k = int(uniq.size)
        self._lut_rgba = np.stack(
            [(uniq >> 24) & 255, (uniq >> 16) & 255,
             (uniq >> 8) & 255, uniq & 255], axis=1).astype(np.int64)

    # -------------------------------------------------------------------
    # Отрисовка разреза
    # -------------------------------------------------------------------

    def _build_mesh(self, pd_, quads):
        px, pz, val = quads
        m = val.shape[0]

        vmin, vmax = pd_.v_min, pd_.v_max
        rng = vmax - vmin if vmax > vmin else 1.0
        t = np.clip((val - vmin) / rng, 0.0, 1.0)
        li = (np.nan_to_num(t) * (LUT_SIZE - 1) + 0.5).astype(np.int32)
        cid = self._lut_cid[li]

        order = np.argsort(cid, kind='stable')
        counts = np.bincount(cid, minlength=self._lut_k)
        bounds = np.concatenate(([0], np.cumsum(counts)))

        # closed quad + NaN separator -> one sub-path per quad
        xo = np.empty((m, 6))
        zo = np.empty((m, 6))
        xo[:, :4] = px[order]
        zo[:, :4] = pz[order]
        xo[:, 4] = xo[:, 0]
        zo[:, 4] = zo[:, 0]
        xo[:, 5] = np.nan
        zo[:, 5] = np.nan

        parts = []
        for k in np.flatnonzero(counts):
            a, b = bounds[k], bounds[k + 1]
            path = arrayToQPath(xo[a:b].ravel(), zo[a:b].ravel(),
                                connect='finite')
            c = self._lut_rgba[k]
            color = QtGui.QColor(int(c[0]), int(c[1]), int(c[2]), int(c[3]))
            pen = QtGui.QPen(color)
            pen.setWidthF(0.5)
            pen.setCosmetic(False)
            parts.append((path, pen, QtGui.QBrush(color)))

        brect = QtCore.QRectF()
        with np.errstate(all='ignore'):
            x_min, x_max = np.nanmin(px), np.nanmax(px)
            z_min, z_max = np.nanmin(pz), np.nanmax(pz)
        if np.isfinite([x_min, x_max, z_min, z_max]).all():
            brect = QtCore.QRectF(float(x_min), float(z_min),
                                  float(x_max - x_min), float(z_max - z_min))
        return FrameMeshItem(parts, brect)

    def _build_boundaries(self, pd_, fr):
        """Layer top/bottom lines of the frame as one NaN-separated curve."""
        st, en = fr.st, fr.en
        n = (en - st).tolist()
        firsts = (pd_.zt[st].tolist(), pd_.zb[st].tolist())
        lasts = (pd_.zt[en - 1].tolist(), pd_.zb[en - 1].tolist())
        z_arrs = (pd_.zt, pd_.zb)
        st_l = st.tolist()
        en_l = en.tolist()

        seen = set()
        xs = []
        zs = []
        gap = np.array([np.nan])
        for s in range(len(st_l)):
            for w in (0, 1):
                key = (firsts[w][s], lasts[w][s], n[s])
                if key in seen:
                    continue
                seen.add(key)
                xs.append(pd_.x[st_l[s]:en_l[s]])
                zs.append(z_arrs[w][st_l[s]:en_l[s]])
                xs.append(gap)
                zs.append(gap)
        if not xs:
            return None
        curve = pg.PlotCurveItem(
            np.concatenate(xs), np.concatenate(zs),
            pen=pg.mkPen(color=(40, 40, 40, 180), width=1.5),
            connect='finite')
        curve.setZValue(1)
        return curve

    def _build_frame(self, pd_, k, mode):
        t0 = time.perf_counter()
        fr = pd_.frame(k)
        items = []
        cost = 0
        quads = build_quads(pd_, fr, mode)
        if quads is not None:
            cost = quads[2].shape[0]
            items.append(self._build_mesh(pd_, quads))
        curve = self._build_boundaries(pd_, fr)
        if curve is not None:
            items.append(curve)
        self._last_build_s = time.perf_counter() - t0
        return items, cost

    def _frame_items(self, prop, k, mode):
        key = (prop, k, mode)
        ent = self._cache.get(key)
        if ent is not None:
            self._cache.move_to_end(key)
            return ent[0]
        items, cost = self._build_frame(self._get_pdata(prop), k, mode)
        self._cache[key] = (items, cost)
        self._cache_cost += cost
        while len(self._cache) > 1 and (
                len(self._cache) > CACHE_MAX_FRAMES
                or self._cache_cost > CACHE_MAX_QUADS):
            _, (_, c) = self._cache.popitem(last=False)
            self._cache_cost -= c
        return items

    def _remove_shown(self):
        for it in self._shown:
            self.plot_widget.removeItem(it)
        self._shown = []

    def _draw(self):
        self._remove_shown()
        self._draw_token += 1

        if not self.current_times:
            return

        prop = self.properties[self.prop_combo.currentIndex()]
        k = self.time_index
        unit = self.units.get(prop, '')
        pd_ = self._get_pdata(prop)
        mode = 'interp' if self.profile_mode.currentIndex() == 1 else 'aver'

        items = self._frame_items(prop, k, mode)
        for it in items:
            self.plot_widget.addItem(it)
        self._shown = list(items)

        self._frame = pd_.frame(k)
        self._current_unit = unit

        self.colorbar.set_range(pd_.v_min, pd_.v_max, unit, self._cmap)

        _fix_y_range(self.plot_widget, self._fixed_z_min, self._fixed_z_max)
        self.plot_widget.enableAutoRange(axis='x')

        self._update_profile()
        self._schedule_prefetch(prop, k, mode)

    def _schedule_prefetch(self, prop, k, mode):
        if self._last_build_s > PREFETCH_MAX_BUILD_S:
            return
        cands = [kk for kk in (k + 1, k - 1)
                 if 0 <= kk < len(self.current_times)]
        token = self._draw_token
        QtCore.QTimer.singleShot(
            0, lambda: self._prefetch(token, prop, mode, cands))

    def _prefetch(self, token, prop, mode, cands):
        """Build neighbouring frames while idle so that stepping is instant."""
        if token != self._draw_token:
            return
        while cands:
            kk = cands.pop(0)
            if (prop, kk, mode) not in self._cache:
                self._frame_items(prop, kk, mode)
                break
        if cands and self._last_build_s <= PREFETCH_MAX_BUILD_S:
            QtCore.QTimer.singleShot(
                0, lambda: self._prefetch(token, prop, mode, cands))

    # -------------------------------------------------------------------
    # 1D профиль по линии среза
    # -------------------------------------------------------------------

    def _update_profile(self, _=None):
        """Строит вертикальный профиль свойства при фиксированном X."""
        fr = self._frame
        if fr is None:
            self._prof_curve.setData(np.empty(0), np.empty(0))
            self._prof_lines.set_zs(np.empty(0))
            return

        pd_ = fr.pd
        sx = float(self._slice_line.value())
        unit = self._current_unit
        if unit != self._prof_unit:
            self.profile_widget.setLabel('bottom', f'Value [{unit}]')
            self._prof_unit = unit

        near, _, _, inside = nearest_rows(pd_.x, fr.st, fr.en, sx)
        rr = near[inside]
        zt = pd_.zt[rr]
        zb = pd_.zb[rr]
        if self.profile_mode.currentIndex() == 0:
            v0 = v1 = pd_.va[rr]
        else:
            v0 = pd_.vt[rr]
            v1 = pd_.vb[rr]

        xs = np.empty(2 * rr.size)
        ys = np.empty(2 * rr.size)
        xs[0::2] = v0
        xs[1::2] = v1
        ys[0::2] = zt
        ys[1::2] = zb
        self._prof_curve.setData(xs, ys, connect='pairs')

        zz = np.concatenate((zt, zb))
        if zz.size:
            _, first = np.unique(np.round(zz, 1), return_index=True)
            zz = zz[first]
        self._prof_lines.set_zs(zz)

        _fix_y_range(self.profile_widget, self._fixed_z_min, self._fixed_z_max)

    # -------------------------------------------------------------------
    # Hover: координаты и значение под курсором
    # -------------------------------------------------------------------

    def _hover_hit(self, fr, mx, mz):
        pd_ = fr.pd
        x = pd_.x
        st, en = fr.st, fr.en
        near, left, nxt, inside = nearest_rows(x, st, en, mx)

        x0 = x[left]
        dx = x[nxt] - x0
        ok = dx != 0
        frac = np.where(ok, (mx - x0) / np.where(ok, dx, 1.0), 0.0)
        zt = pd_.zt[left] + frac * (pd_.zt[nxt] - pd_.zt[left])
        zb = pd_.zb[left] + frac * (pd_.zb[nxt] - pd_.zb[left])

        hit = (inside & (np.minimum(zt, zb) <= mz)
               & (mz <= np.maximum(zt, zb)))
        cand = np.flatnonzero(hit)
        if cand.size == 0:
            return None
        j = cand[np.argmin(fr.dict_rank[cand])]
        return pd_.names[fr.layer[j]], pd_.va[near[j]]

    def _set_info(self, text):
        if text != self._last_hover_text:
            self._last_hover_text = text
            self.info_label.setText(text)

    def _on_mouse_moved(self, pos):
        vb = self.plot_widget.plotItem.vb
        if not vb.sceneBoundingRect().contains(pos):
            if self._cross_visible:
                self._vline.setVisible(False)
                self._hline.setVisible(False)
                self._cross_visible = False
            self._set_info('')
            return

        mouse_pt = vb.mapSceneToView(pos)
        mx, mz = mouse_pt.x(), mouse_pt.y()

        self._vline.setPos(mx)
        self._hline.setPos(mz)
        if not self._cross_visible:
            self._vline.setVisible(True)
            self._hline.setVisible(True)
            self._cross_visible = True

        hit = self._hover_hit(self._frame, mx, mz) if self._frame else None

        unit = self._current_unit
        if hit is not None:
            self._set_info(
                f'X = {mx:.1f} m  |  Depth = {mz:.1f} m  |  '
                f'{hit[0]}  |  Value = {hit[1]:.6g} {unit}'
            )
        else:
            self._set_info(f'X = {mx:.1f} m  |  Depth = {mz:.1f} m')

    # -------------------------------------------------------------------
    # Burial Plot (диаграмма погружения)
    # -------------------------------------------------------------------

    def _show_burial_plot(self):
        """Строит burial plot при текущем X (позиция красной линии среза)."""
        mpl = _import_mpl()
        if mpl is None:
            QtWidgets.QMessageBox.warning(
                self, 'Ошибка',
                'Для burial plot нужен matplotlib:\n  pip install matplotlib'
            )
            return
        plt, LinearSegmentedColormap, LineCollection = mpl

        if self._frame is None:
            return

        sx = self._slice_line.value()
        prop = self.properties[self.prop_combo.currentIndex()]
        unit = self.units.get(prop, '')
        pd_ = self._get_pdata(prop)

        res = burial_field(pd_, sx)
        if res is None:
            return
        ages, depth_grid, field, lines, z_min_g, z_max_g = res

        vmin, vmax = self._fixed_v_min, self._fixed_v_max
        if not vmax > vmin:
            vmax = vmin + 1.0

        cmap_mpl = LinearSegmentedColormap.from_list(
            'petromod',
            [(p, tuple(v / 255.0 for v in c)) for p, c in _CMAP_STOPS],
        )

        fig, ax = plt.subplots(figsize=(11, 6))

        levels = np.linspace(vmin, vmax, 21)
        cf = ax.contourf(
            ages, depth_grid,
            np.ma.masked_invalid(field.T),
            levels=levels, cmap=cmap_mpl, extend='both',
        )

        ax.add_collection(LineCollection(lines, colors='k', linewidths=0.8,
                                         zorder=4))

        ax.set_xlim(ages.max(), ages.min())
        ax.set_ylim(z_max_g, z_min_g)

        ax.set_xlabel('Возраст, млн лет')
        ax.set_ylabel('Глубина, м')
        ax.set_title(f'{prop} — Burial Plot при X = {sx:.0f} m')

        cbar = fig.colorbar(cf, ax=ax, pad=0.01, fraction=0.046)
        cbar.set_label(f'{prop} [{unit}]')

        fig.tight_layout()
        plt.show()


# -----------------------------------------------------------------------
# Запуск
# -----------------------------------------------------------------------

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    app.setStyle('Fusion')
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(45, 45, 45))
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(220, 220, 220))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(35, 35, 35))
    palette.setColor(QtGui.QPalette.Text, QtGui.QColor(220, 220, 220))
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(55, 55, 55))
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(220, 220, 220))
    app.setPalette(palette)

    viewer = BasinViewer()
    viewer.show()
    sys.exit(app.exec_())

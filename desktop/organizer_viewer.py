# Ported from the organizer-provided basin_viewer.py; rendering algorithms retained.
"""
Визуализатор 2D-разреза бассейна.

Зависимости: PyQt5, pyqtgraph, pandas, numpy, openpyxl
"""

import sys
import os
import numpy as np
import pandas as pd

from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

try:
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# -----------------------------------------------------------------------
# Определение папки, где лежит exe / скрипт
# -----------------------------------------------------------------------

def get_app_dir():
    """
    Возвращает папку, в которой лежит exe (если приложение собрано
    PyInstaller-ом) или сам .py файл (при запуске обычным python).

    Никаких абсолютных путей в коде — только эта функция решает,
    где искать данные.
    """
    if getattr(sys, 'frozen', False):
        # Запущено как exe, собранный PyInstaller-ом
        return os.path.dirname(sys.executable)
    else:
        # Запущено как обычный .py скрипт
        return os.path.dirname(os.path.abspath(__file__))


# Обязательные столбцы у файла-«свойства» (Porosity, Lithology, Depth, ...).
# Если xlsx их не содержит (например, lithotype_properties.xlsx — это
# совсем другая таблица), файл пропускается, а не ломает программу.
_REQUIRED_COLUMNS = {
    'X', 'Time_Ma', 'Z_Top [m]', 'Z_Bot [m]', 'Z_Aver [m]',
}


def _sheet_matches_schema(df):
    """Проверяет, что лист xlsx имеет структуру данных свойства слоя."""
    cols = set(df.columns)
    if not _REQUIRED_COLUMNS.issubset(cols):
        return False
    # Плюс должны быть Top/Bot/Aver с единицами измерения в квадратных скобках
    has_top = any(c.startswith('Top') and '[' in c for c in df.columns)
    has_bot = any(c.startswith('Bot') and '[' in c for c in df.columns)
    has_aver = any(c.startswith('Aver') and '[' in c for c in df.columns)
    return has_top and has_bot and has_aver


# -----------------------------------------------------------------------
# Графический элемент: один слой как набор цветных четырёхугольников
# -----------------------------------------------------------------------

class LayerMeshItem(pg.GraphicsObject):
    """
    Рисует один слой разреза: полоса между Z_Top и Z_Bot,
    закрашенная по значению свойства через colormap.
    Использует QPicture для кэширования — перерисовка мгновенная.

    mode='aver':   один цвет на ячейку (значение Aver)
    mode='interp': вертикальная интерполяция Top→Bot (n_sub подъячеек)
    """

    def __init__(self, x, z_top, z_bot, vals_top, vals_bot, vals_aver,
                 cmap, vmin, vmax, mode='aver', n_sub=3):
        super().__init__()
        self._picture = QtGui.QPicture()
        self._brect = QtCore.QRectF()
        self._generate(x, z_top, z_bot, vals_top, vals_bot, vals_aver,
                       cmap, vmin, vmax, mode, n_sub)

    def _generate(self, x, z_top, z_bot, vals_top, vals_bot, vals_aver,
                  cmap, vmin, vmax, mode, n_sub):
        n = len(x)
        if n < 2:
            return

        painter = QtGui.QPainter(self._picture)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, False)

        rng = vmax - vmin if vmax > vmin else 1.0

        for i in range(n - 1):
            if mode == 'interp' and n_sub > 1:
                # Интерполяция Top→Bot: n_sub горизонтальных полосок
                vt_l, vb_l = vals_top[i], vals_bot[i]
                vt_r, vb_r = vals_top[i + 1], vals_bot[i + 1]

                for s in range(n_sub):
                    f0 = s / n_sub
                    f1 = (s + 1) / n_sub

                    # Z координаты подъячейки
                    z0_l = z_top[i] + f0 * (z_bot[i] - z_top[i])
                    z1_l = z_top[i] + f1 * (z_bot[i] - z_top[i])
                    z0_r = z_top[i+1] + f0 * (z_bot[i+1] - z_top[i+1])
                    z1_r = z_top[i+1] + f1 * (z_bot[i+1] - z_top[i+1])

                    # Значение — среднее по подъячейке
                    fm = (f0 + f1) / 2.0
                    val = (vt_l + fm * (vb_l - vt_l)
                           + vt_r + fm * (vb_r - vt_r)) / 2.0

                    t = np.clip((val - vmin) / rng, 0.0, 1.0)
                    rgba = cmap.map(t, mode='byte')
                    color = QtGui.QColor(int(rgba[0]), int(rgba[1]),
                                         int(rgba[2]), int(rgba[3]))
                    painter.setBrush(color)
                    painter.setPen(pg.mkPen(color, width=0.5,
                                            cosmetic=False))

                    poly = QtGui.QPolygonF([
                        QtCore.QPointF(x[i],   z0_l),
                        QtCore.QPointF(x[i+1], z0_r),
                        QtCore.QPointF(x[i+1], z1_r),
                        QtCore.QPointF(x[i],   z1_l),
                    ])
                    painter.drawPolygon(poly)
            else:
                # Режим Aver: один цвет на ячейку
                val = vals_aver[i]
                t = np.clip((val - vmin) / rng, 0.0, 1.0)
                rgba = cmap.map(t, mode='byte')
                color = QtGui.QColor(int(rgba[0]), int(rgba[1]),
                                     int(rgba[2]), int(rgba[3]))
                painter.setBrush(color)
                painter.setPen(pg.mkPen(color, width=0.5,
                                        cosmetic=False))

                poly = QtGui.QPolygonF([
                    QtCore.QPointF(x[i],     z_top[i]),
                    QtCore.QPointF(x[i + 1], z_top[i + 1]),
                    QtCore.QPointF(x[i + 1], z_bot[i + 1]),
                    QtCore.QPointF(x[i],     z_bot[i]),
                ])
                painter.drawPolygon(poly)

        painter.end()

        x_min, x_max = x.min(), x.max()
        z_all_min = min(z_top.min(), z_bot.min())
        z_all_max = max(z_top.max(), z_bot.max())
        self._brect = QtCore.QRectF(
            x_min, z_all_min,
            x_max - x_min, z_all_max - z_all_min,
        )

    def paint(self, painter, *args):
        painter.drawPicture(0, 0, self._picture)

    def boundingRect(self):
        return self._brect


# -----------------------------------------------------------------------
# Виджет цветовой шкалы (colorbar)
# -----------------------------------------------------------------------

class ColorBarWidget(QtWidgets.QWidget):
    """Вертикальная цветовая шкала с подписями min/max."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(96)
        self.cmap = None
        self.vmin = 0.0
        self.vmax = 1.0
        self.unit = ''

    def set_range(self, vmin, vmax, unit, cmap):
        self.vmin = vmin
        self.vmax = vmax
        self.unit = unit
        self.cmap = cmap
        self.update()

    def paintEvent(self, event):
        if self.cmap is None:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        bar_x = 10
        bar_w = 25
        bar_top = 30
        bar_bot = h - 30
        bar_h = bar_bot - bar_top

        # Рисуем градиент
        for iy in range(bar_h):
            t = 1.0 - iy / bar_h  # сверху max, снизу min
            rgba = self.cmap.map(t, mode='byte')
            color = QtGui.QColor(int(rgba[0]), int(rgba[1]),
                                 int(rgba[2]), int(rgba[3]))
            painter.setPen(color)
            painter.drawLine(bar_x, bar_top + iy, bar_x + bar_w, bar_top + iy)

        # Рамка
        painter.setPen(QtGui.QColor(180, 180, 180))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawRect(bar_x, bar_top, bar_w, bar_h)

        # Подписи
        painter.setPen(QtGui.QColor(220, 220, 220))
        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)

        txt_x = bar_x + bar_w + 5
        painter.drawText(txt_x, bar_top + 12, f'{self.vmax:.4g}')
        painter.drawText(txt_x, bar_bot, f'{self.vmin:.4g}')

        # Промежуточные отметки
        n_ticks = 4
        for i in range(1, n_ticks):
            frac = i / n_ticks
            yy = bar_top + int(frac * bar_h)
            val = self.vmax - frac * (self.vmax - self.vmin)
            painter.drawLine(bar_x + bar_w - 3, yy, bar_x + bar_w, yy)
            painter.drawText(txt_x, yy + 4, f'{val:.4g}')

        # Единицы
        if self.unit:
            painter.drawText(bar_x, bar_top - 8, self.unit)

        painter.end()


# -----------------------------------------------------------------------
# Главное окно
# -----------------------------------------------------------------------

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


class BasinViewer(QtWidgets.QWidget):

    def __init__(self, directory=None, loaded=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('2D Basin Viewer')
        self.resize(1200, 700)

        # Состояние
        # Папка с данными — всегда рядом с exe/скриптом, нигде не хардкожена
        self.xlsx_dir = str(directory) if directory is not None else get_app_dir()
        self.data = {}          # prop_name -> {layer_name -> DataFrame}
        self.properties = []    # список имён свойств
        self.units = {}         # prop_name -> unit string
        self.time_index = 0
        self.current_times = []
        self._active_layers = {}  # текущие слои на экране (для hover)

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

    # -------------------------------------------------------------------
    # UI
    # -------------------------------------------------------------------

    def _init_ui(self):
        central = self

        # ----- Верхняя панель -----
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

        # Режим профиля
        toolbar.addWidget(QtWidgets.QLabel('Профиль:'))
        self.profile_mode = QtWidgets.QComboBox()
        self.profile_mode.addItems(['Aver', 'Top→Bot'])
        self.profile_mode.setFixedWidth(100)
        self.profile_mode.currentIndexChanged.connect(self._on_display_mode_changed)
        toolbar.addWidget(self.profile_mode)

        toolbar.addSpacing(8)

        # Кнопка Burial Plot
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
        # Отключаем автопреобразование единиц (м → км)
        for axis_name in ('bottom', 'left'):
            ax = self.plot_widget.getAxis(axis_name)
            ax.enableAutoSIPrefix(False)

        # Перекрестие
        self._vline = pg.InfiniteLine(angle=90, pen=pg.mkPen('y', width=1,
                                      style=QtCore.Qt.DashLine))
        self._hline = pg.InfiniteLine(angle=0, pen=pg.mkPen('y', width=1,
                                      style=QtCore.Qt.DashLine))
        self._vline.setVisible(False)
        self._hline.setVisible(False)
        self.plot_widget.addItem(self._vline, ignoreBounds=True)
        self.plot_widget.addItem(self._hline, ignoreBounds=True)

        # Подключаем мышь
        self.plot_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)

        # ----- Линия среза (перетаскиваемая) -----
        self._slice_line = pg.InfiniteLine(
            angle=90, movable=True,
            pen=pg.mkPen(color=(255, 80, 80), width=2),
            hoverPen=pg.mkPen(color=(255, 120, 120), width=3),
        )
        self._slice_line.sigPositionChanged.connect(self._update_profile)
        self.plot_widget.addItem(self._slice_line, ignoreBounds=True)

        # ----- 1D профиль (правая панель) -----
        self.profile_widget = pg.PlotWidget()
        self.profile_widget.setLabel('bottom', 'Value')
        self.profile_widget.setLabel('left', 'Depth, m')
        self.profile_widget.invertY(True)
        self.profile_widget.showGrid(x=True, y=True, alpha=0.2)
        self.profile_widget.setMinimumWidth(160)
        self.profile_widget.setMaximumWidth(240)
        for axis_name in ('bottom', 'left'):
            self.profile_widget.getAxis(axis_name).enableAutoSIPrefix(False)

        # ----- Info label (внизу) -----
        self.info_label = QtWidgets.QLabel('')
        self.info_label.setStyleSheet(
            'color: #cccccc; font-family: Consolas, monospace; font-size: 11px;'
            'padding: 3px 6px;'
        )

        # ----- Colorbar -----
        self.colorbar = ColorBarWidget()

        # ----- Компоновка -----
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
        """Собирает все уникальные временные шаги для свойства.
        Сортировка от старого к молодому (200 Ma → 0 Ma),
        чтобы ▶ шёл «вперёд по времени» к настоящему.
        """
        times = set()
        for df in self.data[prop_name].values():
            times.update(df.iloc[:, 1].unique())
        return sorted(times, reverse=True)

    def _on_property_changed(self):
        prop = self.properties[self.prop_combo.currentIndex()]

        # Запоминаем текущее время, чтобы не сбрасывать
        old_time = (self.current_times[self.time_index]
                    if self.current_times else None)

        self.current_times = self._gather_times(prop)
        self._compute_fixed_ranges(prop)

        # Ищем ближайший доступный временной шаг
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
        """Вычисляет глобальные диапазоны глубин и значений
        по ВСЕМ временным шагам — чтобы шкалы не прыгали."""
        z_min_global = float('inf')
        z_max_global = float('-inf')
        v_min_global = float('inf')
        v_max_global = float('-inf')
        for df in self.data[prop_name].values():
            z_top = df.iloc[:, 2].values
            z_bot = df.iloc[:, 3].values
            vals = df.iloc[:, 7].values
            z_min_global = min(z_min_global, np.nanmin(z_top), np.nanmin(z_bot))
            z_max_global = max(z_max_global, np.nanmax(z_top), np.nanmax(z_bot))
            v_min_global = min(v_min_global, np.nanmin(vals))
            v_max_global = max(v_max_global, np.nanmax(vals))
        margin = (z_max_global - z_min_global) * 0.03
        self._fixed_z_min = z_min_global - margin
        self._fixed_z_max = z_max_global + margin
        self._fixed_v_min = float(v_min_global)
        self._fixed_v_max = float(v_max_global)

    def _on_display_mode_changed(self):
        """Перерисовка разреза и профиля при смене режима Aver/Top→Bot."""
        self._draw()

    # -------------------------------------------------------------------
    # Палитра
    # -------------------------------------------------------------------

    def _get_colormap(self):
        colors = [
            (0.0,   (  0,   0, 180)),
            (0.15,  (  0,  80, 220)),
            (0.30,  (  0, 180, 220)),
            (0.45,  (  0, 200, 150)),
            (0.55,  ( 50, 210,  50)),
            (0.70,  (150, 220,  30)),
            (0.85,  (220, 220,   0)),
            (1.0,   (255, 255,  50)),
        ]
        return pg.ColorMap([c[0] for c in colors], [c[1] for c in colors])

    # -------------------------------------------------------------------
    # Отрисовка разреза
    # -------------------------------------------------------------------

    def _draw(self):
        self.plot_widget.clear()

        if not self.current_times:
            return

        prop = self.properties[self.prop_combo.currentIndex()]
        time = self.current_times[self.time_index]
        unit = self.units.get(prop, '')
        layers_data = self.data[prop]

        # Собираем слои, существующие в данный момент времени
        active_layers = {}
        for layer_name, df in layers_data.items():
            mask = df.iloc[:, 1] == time
            sub = df.loc[mask]
            if len(sub) > 0:
                active_layers[layer_name] = sub

        if not active_layers:
            return

        # Фиксированный диапазон значений (по всем временным шагам)
        vmin, vmax = self._fixed_v_min, self._fixed_v_max

        cmap = self._get_colormap()

        # Сортируем слои: сначала глубокие (большие номера)
        def layer_sort_key(name):
            parts = name.replace('Layer_', '').split('_')
            return tuple(int(p) for p in parts)

        sorted_layers = sorted(
            active_layers.items(), key=lambda kv: layer_sort_key(kv[0]),
            reverse=True,  # глубокие первыми (рисуются под верхними)
        )

        for layer_name, sub in sorted_layers:
            x = sub.iloc[:, 0].values.astype(np.float64)
            z_top = sub.iloc[:, 2].values.astype(np.float64)
            z_bot = sub.iloc[:, 3].values.astype(np.float64)
            vals_top = sub.iloc[:, 5].values.astype(np.float64)
            vals_bot = sub.iloc[:, 6].values.astype(np.float64)
            vals_aver = sub.iloc[:, 7].values.astype(np.float64)

            draw_mode = 'interp' if self.profile_mode.currentIndex() == 1 else 'aver'
            item = LayerMeshItem(x, z_top, z_bot,
                                 vals_top, vals_bot, vals_aver,
                                 cmap, vmin, vmax,
                                 mode=draw_mode, n_sub=3)
            self.plot_widget.addItem(item)

        # Границы между слоями — тонкие линии поверх заливки
        boundary_pen = pg.mkPen(color=(40, 40, 40, 180), width=1.5)
        drawn_boundaries = set()
        for layer_name, sub in sorted_layers:
            x = sub.iloc[:, 0].values.astype(np.float64)
            z_top = sub.iloc[:, 2].values.astype(np.float64)
            z_bot = sub.iloc[:, 3].values.astype(np.float64)

            # Рисуем кровлю (top) и подошву (bot), дедупликация по hash
            for z_arr in (z_top, z_bot):
                key = (z_arr[0], z_arr[-1], len(z_arr))
                if key not in drawn_boundaries:
                    drawn_boundaries.add(key)
                    curve = pg.PlotCurveItem(x, z_arr, pen=boundary_pen)
                    self.plot_widget.addItem(curve)

        # Сохраняем для hover
        self._active_layers = active_layers
        self._current_unit = unit

        # Перекрестие и линия среза (добавляем после clear)
        self.plot_widget.addItem(self._vline, ignoreBounds=True)
        self.plot_widget.addItem(self._hline, ignoreBounds=True)
        self.plot_widget.addItem(self._slice_line, ignoreBounds=True)

        # Обновляем colorbar
        self.colorbar.set_range(vmin, vmax, unit, cmap)

        # Фиксированная шкала глубин, автомасштаб по X
        self.plot_widget.setYRange(self._fixed_z_min, self._fixed_z_max,
                                   padding=0)
        self.plot_widget.enableAutoRange(axis='x')

        # Обновляем 1D профиль
        self._update_profile()

    # -------------------------------------------------------------------
    # 1D профиль по линии среза
    # -------------------------------------------------------------------

    def _update_profile(self, _=None):
        """Строит вертикальный профиль свойства при фиксированном X."""
        self.profile_widget.clear()

        if not self._active_layers:
            return

        sx = self._slice_line.value()
        unit = getattr(self, '_current_unit', '')
        mode = self.profile_mode.currentIndex()  # 0=Aver, 1=Top→Bot
        self.profile_widget.setLabel('bottom', f'Value [{unit}]')

        def layer_sort_key(name):
            parts = name.replace('Layer_', '').split('_')
            return tuple(int(p) for p in parts)

        for layer_name, sub in sorted(
            self._active_layers.items(), key=lambda kv: layer_sort_key(kv[0])
        ):
            x = sub.iloc[:, 0].values
            z_top = sub.iloc[:, 2].values
            z_bot = sub.iloc[:, 3].values
            vals_top = sub.iloc[:, 5].values
            vals_bot = sub.iloc[:, 6].values
            vals_aver = sub.iloc[:, 7].values

            if sx < x[0] or sx > x[-1]:
                continue

            idx = np.searchsorted(x, sx, side='right') - 1
            idx = np.clip(idx, 0, len(x) - 1)
            nearest = idx if (idx >= len(x) - 1 or
                              sx - x[idx] <= x[min(idx+1, len(x)-1)] - sx
                              ) else idx + 1

            zt = z_top[nearest]
            zb = z_bot[nearest]

            if mode == 0:
                # Aver: постоянное значение от z_top до z_bot
                val = vals_aver[nearest]
                self.profile_widget.plot(
                    [val, val], [zt, zb],
                    pen=pg.mkPen(color=(100, 200, 255), width=2),
                )
            else:
                # Top→Bot: линейная интерполяция
                vt = vals_top[nearest]
                vb = vals_bot[nearest]
                self.profile_widget.plot(
                    [vt, vb], [zt, zb],
                    pen=pg.mkPen(color=(100, 200, 255), width=2),
                )

        # Горизонтальные линии на границах слоёв
        drawn_z = set()
        for layer_name, sub in self._active_layers.items():
            x = sub.iloc[:, 0].values
            z_top = sub.iloc[:, 2].values
            z_bot = sub.iloc[:, 3].values

            if sx < x[0] or sx > x[-1]:
                continue

            idx = np.searchsorted(x, sx, side='right') - 1
            idx = np.clip(idx, 0, len(x) - 1)
            nearest = idx if (idx >= len(x) - 1 or
                              sx - x[idx] <= x[min(idx+1, len(x)-1)] - sx
                              ) else idx + 1

            for z in (z_top[nearest], z_bot[nearest]):
                zr = round(z, 1)
                if zr not in drawn_z:
                    drawn_z.add(zr)
                    self.profile_widget.addItem(
                        pg.InfiniteLine(pos=z, angle=0,
                                        pen=pg.mkPen((80, 80, 80), width=0.5))
                    )

        # Фиксированная шкала глубин
        self.profile_widget.setYRange(self._fixed_z_min, self._fixed_z_max,
                                       padding=0)

    # -------------------------------------------------------------------
    # Hover: координаты и значение под курсором
    # -------------------------------------------------------------------

    def _on_mouse_moved(self, pos):
        """Обработка движения мыши по графику."""
        vb = self.plot_widget.plotItem.vb
        if not vb.sceneBoundingRect().contains(pos):
            self._vline.setVisible(False)
            self._hline.setVisible(False)
            self.info_label.setText('')
            return

        mouse_pt = vb.mapSceneToView(pos)
        mx, mz = mouse_pt.x(), mouse_pt.y()

        self._vline.setPos(mx)
        self._hline.setPos(mz)
        self._vline.setVisible(True)
        self._hline.setVisible(True)

        # Ищем слой, в который попадает курсор
        hit_layer = None
        hit_value = None

        for layer_name, sub in self._active_layers.items():
            x = sub.iloc[:, 0].values
            z_top = sub.iloc[:, 2].values
            z_bot = sub.iloc[:, 3].values
            vals = sub.iloc[:, 7].values

            if mx < x[0] or mx > x[-1]:
                continue

            # Находим индекс ближайшей X-точки слева
            idx = np.searchsorted(x, mx, side='right') - 1
            idx = np.clip(idx, 0, len(x) - 1)

            # Интерполяция Z_top и Z_bot по X (для попадания)
            if idx < len(x) - 1:
                frac = (mx - x[idx]) / (x[idx + 1] - x[idx]) \
                    if x[idx + 1] != x[idx] else 0.0
                zt = z_top[idx] + frac * (z_top[idx + 1] - z_top[idx])
                zb = z_bot[idx] + frac * (z_bot[idx + 1] - z_bot[idx])
            else:
                zt, zb = z_top[idx], z_bot[idx]

            # Значение — ближайший узел (без интерполяции,
            # чтобы категориальные свойства не смешивались)
            nearest = idx if (idx >= len(x) - 1 or
                              mx - x[idx] <= x[min(idx+1, len(x)-1)] - mx
                              ) else idx + 1
            val = vals[nearest]

            # Попадает ли mz между кровлей и подошвой?
            z_lo, z_hi = min(zt, zb), max(zt, zb)
            if z_lo <= mz <= z_hi:
                hit_layer = layer_name
                hit_value = val
                break

        unit = getattr(self, '_current_unit', '')
        if hit_layer is not None:
            self.info_label.setText(
                f'X = {mx:.1f} m  |  Depth = {mz:.1f} m  |  '
                f'{hit_layer}  |  Value = {hit_value:.6g} {unit}'
            )
        else:
            self.info_label.setText(
                f'X = {mx:.1f} m  |  Depth = {mz:.1f} m'
            )

    # -------------------------------------------------------------------
    # Burial Plot (диаграмма погружения)
    # -------------------------------------------------------------------

    def _show_burial_plot(self):
        """Строит burial plot при текущем X (позиция красной линии среза)."""
        if not HAS_MPL:
            QtWidgets.QMessageBox.warning(
                self, 'Ошибка',
                'Для burial plot нужен matplotlib:\n  pip install matplotlib'
            )
            return

        if not self._active_layers:
            return

        sx = self._slice_line.value()
        prop = self.properties[self.prop_combo.currentIndex()]
        unit = self.units.get(prop, '')
        layers_data = self.data[prop]

        def layer_sort_key(name):
            parts = name.replace('Layer_', '').split('_')
            return tuple(int(p) for p in parts)

        # Собираем данные: для каждого временного шага —
        # границы слоёв и значения при X = sx
        all_times = sorted(self._gather_times(prop))  # молодые→старые
        time_layers = {}  # time -> [(z_top, z_bot, val_top, val_bot, layer_name)]

        for t in all_times:
            entries = []
            for layer_name, df in layers_data.items():
                sub = df[df.iloc[:, 1] == t]
                if len(sub) == 0:
                    continue
                x = sub.iloc[:, 0].values
                if sx < x[0] or sx > x[-1]:
                    continue

                idx = np.searchsorted(x, sx, side='right') - 1
                idx = np.clip(idx, 0, len(x) - 1)
                nearest = idx if (idx >= len(x) - 1 or
                                  sx - x[idx] <= x[min(idx+1, len(x)-1)] - sx
                                  ) else idx + 1

                zt = sub.iloc[nearest, 2]
                zb = sub.iloc[nearest, 3]
                vt = sub.iloc[nearest, 5]
                vb = sub.iloc[nearest, 6]
                entries.append((zt, zb, vt, vb, layer_name))

            entries.sort(key=lambda e: e[0])  # по глубине
            time_layers[t] = entries

        if not time_layers:
            return

        # Определяем глобальные диапазоны
        all_z = [e[0] for t in time_layers for e in time_layers[t]]
        all_z += [e[1] for t in time_layers for e in time_layers[t]]
        z_min_g, z_max_g = min(all_z), max(all_z)

        vmin, vmax = self._fixed_v_min, self._fixed_v_max

        # Сетка для contourf
        n_z = 300
        depth_grid = np.linspace(z_min_g, z_max_g, n_z)
        ages = np.array(sorted(time_layers.keys()))

        # Мелкая временная сетка (шаг ~1 Ma) для гладких границ
        age_fine = np.arange(ages.min(), ages.max() + 0.5, 1.0)
        # Добавляем оригинальные точки чтобы не потерять
        age_fine = np.unique(np.concatenate([age_fine, ages]))
        age_fine.sort()

        field = np.full((len(age_fine), n_z), np.nan)

        # Для каждого слоя интерполируем границы и значения по времени
        all_layer_names = set()
        for entries in time_layers.values():
            for e in entries:
                all_layer_names.add(e[4])

        for ln in all_layer_names:
            # Собираем точки (time, zt, zb, vt, vb) для этого слоя
            pts = []
            for t in ages:
                for zt, zb, vt, vb, name in time_layers[t]:
                    if name == ln:
                        pts.append((t, zt, zb, vt, vb))
            if not pts:
                continue

            pts.sort(key=lambda p: p[0])
            t_pts = np.array([p[0] for p in pts])
            zt_pts = np.array([p[1] for p in pts])
            zb_pts = np.array([p[2] for p in pts])
            vt_pts = np.array([p[3] for p in pts])
            vb_pts = np.array([p[4] for p in pts])

            # Интерполяция на мелкую сетку (только в диапазоне существования слоя)
            mask_t = (age_fine >= t_pts.min()) & (age_fine <= t_pts.max())
            t_interp = age_fine[mask_t]

            zt_i = np.interp(t_interp, t_pts, zt_pts)
            zb_i = np.interp(t_interp, t_pts, zb_pts)
            vt_i = np.interp(t_interp, t_pts, vt_pts)
            vb_i = np.interp(t_interp, t_pts, vb_pts)

            for j, ti in enumerate(t_interp):
                idx_t = np.searchsorted(age_fine, ti)
                if idx_t >= len(age_fine):
                    continue
                z0, z1 = zt_i[j], zb_i[j]
                v0, v1 = vt_i[j], vb_i[j]
                inside = (depth_grid >= z0) & (depth_grid <= z1)
                if z1 > z0:
                    frac = (depth_grid[inside] - z0) / (z1 - z0)
                    field[idx_t, inside] = v0 + frac * (v1 - v0)
                elif np.any(inside):
                    field[idx_t, inside] = v0

        ages = age_fine  # используем мелкую сетку для отрисовки

        pm_colors = [
            (0.0,   (0/255,   0/255, 180/255)),
            (0.15,  (0/255,  80/255, 220/255)),
            (0.30,  (0/255, 180/255, 220/255)),
            (0.45,  (0/255, 200/255, 150/255)),
            (0.55,  (50/255, 210/255,  50/255)),
            (0.70,  (150/255, 220/255,  30/255)),
            (0.85,  (220/255, 220/255,   0/255)),
            (1.0,   (255/255, 255/255,  50/255)),
        ]
        cmap_mpl = LinearSegmentedColormap.from_list(
            'petromod',
            [(p, c) for p, c in pm_colors],
        )

        # Рисуем contourf
        fig, ax = plt.subplots(figsize=(11, 6))

        n_levels = 21
        levels = np.linspace(vmin, vmax, n_levels)
        cf = ax.contourf(
            ages, depth_grid,
            np.ma.masked_invalid(field.T),
            levels=levels, cmap=cmap_mpl, extend='both',
        )

        # Границы слоёв (линии погружения) — по интерполированным данным
        for ln in sorted(all_layer_names, key=layer_sort_key):
            pts = []
            for t in sorted(time_layers.keys()):
                for zt, zb, vt, vb, name in time_layers[t]:
                    if name == ln:
                        pts.append((t, zt, zb))
            if not pts:
                continue
            pts.sort(key=lambda p: p[0])
            t_pts = np.array([p[0] for p in pts])
            zt_pts = np.array([p[1] for p in pts])
            zb_pts = np.array([p[2] for p in pts])

            t_fine_l = np.linspace(t_pts.min(), t_pts.max(), 200)
            ax.plot(t_fine_l, np.interp(t_fine_l, t_pts, zt_pts),
                    'k-', lw=0.8, zorder=4)
            ax.plot(t_fine_l, np.interp(t_fine_l, t_pts, zb_pts),
                    'k-', lw=0.8, zorder=4)

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

    # Тёмная тема
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
"""Basin Solver: the three screens in the supplied wireframes."""
from __future__ import annotations
import asyncio
import os
from collections import OrderedDict
from pathlib import Path
import sys
import threading

from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import pipeline
import inverse_pipeline
from directory_picker import portal_directory

BG = '#111111'
pg.setConfigOptions(background=BG, foreground='#dddddd', useOpenGL=False)
FORWARD = pipeline.EDITABLE
INVERSE = inverse_pipeline.INVERSE_PARAMETERS
STYLE = '''
QWidget#window { background: transparent; }
QWidget { background: #111111; color: #dddddd; font-family: "DejaVu Sans"; font-size: 12px; }
QFrame#shell { border: 2px solid #cccccc; border-radius: 20px; }
QFrame#title { border: none; border-bottom: 1px solid #bbbbbb; background: transparent; }
QLabel#brand { font-size: 21px; background: transparent; }
QPushButton { border: 1px solid #cccccc; border-radius: 9px; padding: 8px 16px; background: #111111; }
QPushButton:hover { background: #242424; }
QPushButton:checked { background: #262626; }
QPushButton:disabled { color: #696969; border-color: #555555; }
QPushButton#windowControl { border: none; padding: 0; font-size: 21px; border-radius: 0; background: transparent; }
QPushButton#homeAction { border-radius: 16px; font-size: 23px; }
QFrame#sidebar { border: none; border-right: 1px solid #bbbbbb; }
QFrame#panel { border: 1px solid #bbbbbb; border-radius: 18px; }
QFrame#logPanel { border: none; border-top: 1px solid #bbbbbb; }
QLabel#treeTitle, QLabel#logTitle { font-size: 21px; }
QTreeWidget { border: none; outline: none; background: #111111; }
QTreeWidget::item { padding: 6px 2px; }
QTreeWidget::item:selected { background: #292929; }
QTabWidget::pane { border: none; }
QTabBar::tab { border: 1px solid #cccccc; border-radius: 6px; padding: 5px 18px; margin: 4px 3px 4px 0; }
QTabBar::tab:selected { background: #242424; }
QTableWidget { background: #111111; border: none; gridline-color: #aaaaaa; selection-background-color: #292929; }
QTableWidget QLineEdit { border: none; background: #111111; padding: 5px 8px; }
QComboBox { border: 1px solid #777777; border-radius: 4px; padding: 3px 7px; }
QComboBox::drop-down { border: none; width: 18px; }
QPlainTextEdit { border: none; background: #111111; font-family: "DejaVu Sans Mono"; font-size: 10px; }
QScrollBar:vertical { background: #111111; width: 10px; }
QScrollBar::handle:vertical { background: #777777; min-height: 20px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
'''


class Signals(QtCore.QObject):
    result = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(object)
    log = QtCore.pyqtSignal(str)


class Task(QtCore.QRunnable):
    def __init__(self, function):
        super().__init__()
        self.signals = Signals()
        self.function = function
        self.cancelled = threading.Event()

    def run(self):
        try:
            result = self.function(self.signals.log.emit, self.cancelled.is_set)
        except Exception as error:
            self.signals.error.emit(error)
        else:
            self.signals.result.emit(result)


class FolderTree(QtWidgets.QTreeWidget):
    def mousePressEvent(self, event):
        item = self.itemAt(event.pos())
        if item is not None and item.parent() is None and event.pos().x() >= self.visualItemRect(item).left():
            item.setExpanded(not item.isExpanded())
        super().mousePressEvent(event)


class ViewerHost(QtWidgets.QWidget):
    cache = OrderedDict()
    loaded = QtCore.pyqtSignal()

    def __init__(self, log=lambda _: None):
        super().__init__()
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.viewer = None
        self.directory = None
        self.task = None
        self.generation = 0
        self.log = log

    def clear(self):
        while self.layout.count():
            widget = self.layout.takeAt(0).widget()
            if widget:
                widget.deleteLater()
        self.viewer = None

    def load(self, directory, preferred=None):
        directory = Path(directory).resolve()
        signature = (str(directory), tuple((path.name, path.stat().st_size, path.stat().st_mtime_ns)
                     for path in sorted(directory.glob('*.xlsx')) if not path.name.startswith('~$')))
        if self.directory == signature and (self.viewer is not None or self.task is not None):
            return
        if self.task:
            self.task.cancelled.set()
        self.generation += 1
        generation = self.generation
        self.directory = signature
        self.clear()
        loading = QtWidgets.QLabel('Loading…')
        loading.setAlignment(QtCore.Qt.AlignCenter)
        self.layout.addWidget(loading)

        def finish(data):
            if generation != self.generation:
                return
            self.task = None
            self.clear()
            try:
                from organizer_viewer import BasinViewer
                self.viewer = BasinViewer(directory=directory, loaded=data, parent=self)
                self.layout.addWidget(self.viewer)
                if preferred and preferred in self.viewer.properties:
                    self.viewer.prop_combo.setCurrentIndex(self.viewer.properties.index(preferred))
                self.cache[signature] = data
                self.cache.move_to_end(signature)
                while len(self.cache) > 3:
                    self.cache.popitem(last=False)
                self.loaded.emit()
            except Exception as error:
                failure(error)

        def failure(error):
            if generation != self.generation:
                return
            self.task = None
            self.clear()
            text = QtWidgets.QLabel(str(error))
            text.setWordWrap(True)
            text.setAlignment(QtCore.Qt.AlignCenter)
            self.layout.addWidget(text)
            self.log(str(error))

        if signature in self.cache:
            finish(self.cache[signature])
            return
        def work(emit, cancelled):
            from viewer_data import load_directory
            return load_directory(directory, emit, cancelled)
        self.task = Task(work)
        self.task.signals.result.connect(finish)
        self.task.signals.error.connect(failure)
        self.task.signals.log.connect(self.log)
        QtCore.QThreadPool.globalInstance().start(self.task)


class ModePage(QtWidgets.QWidget):
    def __init__(self, window, inverse=False):
        super().__init__()
        self.window = window
        self.inverse = inverse
        self.engine = inverse_pipeline if inverse else pipeline
        self.sources = {}
        self.results = {}
        self.active = None
        self.runner = None
        self.running = False
        self.task = None
        self.picker_task = None
        self.defaults = self.engine.read_defaults()
        body = QtWidgets.QHBoxLayout(self)
        body.setContentsMargins(0, 0, 0, 0); body.setSpacing(0)
        sidebar = QtWidgets.QFrame(); sidebar.setObjectName('sidebar'); sidebar.setFixedWidth(220)
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(12, 10, 8, 10)
        title = QtWidgets.QLabel('Tree'); title.setObjectName('treeTitle')
        sidebar_layout.addWidget(title)
        self.tree = FolderTree(); self.tree.setHeaderHidden(True); self.tree.setExpandsOnDoubleClick(False)
        self.tree.setIndentation(18)
        self.tree.itemClicked.connect(self.select_item)
        sidebar_layout.addWidget(self.tree)
        body.addWidget(sidebar)
        right = QtWidgets.QWidget(); right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0); right_layout.setSpacing(0)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.currentChanged.connect(self.tab_changed)
        self.main = QtWidgets.QWidget(); self.visual = QtWidgets.QWidget()
        self.tabs.addTab(self.main, 'main'); self.tabs.addTab(self.visual, 'visualize')
        right_layout.addWidget(self.tabs, 1)
        main_layout = QtWidgets.QVBoxLayout(self.main)
        main_layout.setContentsMargins(18, 4, 18, 14); main_layout.setSpacing(18)
        table_frame = QtWidgets.QFrame(); table_frame.setObjectName('panel')
        table_layout = QtWidgets.QVBoxLayout(table_frame); table_layout.setContentsMargins(4, 7, 4, 7)
        self.table = QtWidgets.QTableWidget(len(self.engine.EDITABLE), 2)
        self.table.horizontalHeader().hide(); self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.verticalHeader().setDefaultSectionSize(24 if inverse else 30)
        self.editors = {}
        for row, key in enumerate(self.engine.EDITABLE):
            label = QtWidgets.QTableWidgetItem(key)
            label.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            self.table.setItem(row, 0, label)
            if key in self.engine.CHOICES:
                editor = QtWidgets.QComboBox(); editor.addItems(self.engine.CHOICES[key]); editor.setCurrentText(self.defaults[key])
            else:
                editor = QtWidgets.QLineEdit(self.format_value(self.defaults[key]))
            self.table.setCellWidget(row, 1, editor)
            self.editors[key] = editor
        table_layout.addWidget(self.table); main_layout.addWidget(table_frame, 1)
        self.pick_button = QtWidgets.QPushButton('Pick a folder with data')
        self.start_button = QtWidgets.QPushButton('Start')
        self.stop_button = QtWidgets.QPushButton('Stop'); self.stop_button.setEnabled(False)
        for button in (self.pick_button, self.start_button, self.stop_button):
            button.setFixedSize(240, 46); main_layout.addWidget(button, 0, QtCore.Qt.AlignHCenter)
        self.pick_button.clicked.connect(self.pick_folder)
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        visual_layout = QtWidgets.QVBoxLayout(self.visual)
        visual_layout.setContentsMargins(18, 4, 18, 14)
        visual_frame = QtWidgets.QFrame(); visual_frame.setObjectName('panel')
        visual_layout.addWidget(visual_frame)
        visual_inner = QtWidgets.QVBoxLayout(visual_frame); visual_inner.setContentsMargins(12, 12, 12, 12)
        self.viewer = ViewerHost(self.append_log)
        self.advanced = None
        if inverse:
            options = QtWidgets.QHBoxLayout(); options.setContentsMargins(0, 0, 0, 0)
            buttons = QtWidgets.QVBoxLayout(); buttons.setSpacing(5)
            self.simple_button = QtWidgets.QPushButton('Simple'); self.simple_button.setCheckable(True)
            self.advanced_button = QtWidgets.QPushButton('Advanced'); self.advanced_button.setCheckable(True)
            group = QtWidgets.QButtonGroup(self); group.setExclusive(True)
            for button in (self.simple_button, self.advanced_button):
                button.setFixedSize(95, 32); group.addButton(button); buttons.addWidget(button)
            options.addLayout(buttons); options.addStretch()
            visual_inner.addLayout(options)
            self.views = QtWidgets.QStackedWidget(); self.views.addWidget(self.viewer)
            self.advanced = pg.PlotWidget(background=BG); self.advanced.setLabel('left', 'q'); self.advanced.setLabel('bottom', 't, ма')
            self.advanced.setMouseEnabled(x=False, y=False); self.advanced.setMenuEnabled(False)
            self.advanced.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
            self.advanced.getPlotItem().hideButtons(); self.advanced.setFocusPolicy(QtCore.Qt.NoFocus)
            self.advanced.setRange(xRange=(0, 200), yRange=(0, 1), padding=0)
            self.views.addWidget(self.advanced); visual_inner.addWidget(self.views, 1)
            self.simple_button.setChecked(True)
            self.simple_button.clicked.connect(lambda: self.views.setCurrentIndex(0))
            self.advanced_button.clicked.connect(lambda: self.views.setCurrentIndex(1))
        else:
            visual_inner.addWidget(self.viewer, 1)
        log_frame = QtWidgets.QFrame(); log_frame.setObjectName('logPanel'); log_frame.setFixedHeight(145)
        log_layout = QtWidgets.QHBoxLayout(log_frame); log_layout.setContentsMargins(10, 8, 8, 8)
        log_title = QtWidgets.QLabel('LOG:'); log_title.setObjectName('logTitle'); log_title.setAlignment(QtCore.Qt.AlignTop)
        log_layout.addWidget(log_title)
        self.log = QtWidgets.QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumBlockCount(5000)
        self.log.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap); log_layout.addWidget(self.log, 1)
        right_layout.addWidget(log_frame); body.addWidget(right, 1)

    @staticmethod
    def format_value(value):
        try:
            return format(float(value), '.12g')
        except ValueError:
            return value

    def append_log(self, text):
        self.log.appendPlainText(text)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def values(self):
        return {key:editor.currentText() if isinstance(editor, QtWidgets.QComboBox) else editor.text()
                for key,editor in self.editors.items()}

    def add_directory(self, directory):
        directory = Path(directory).resolve()
        if self.running:
            return
        if not directory.is_dir():
            self.append_log(f'Folder does not exist: {directory}'); return
        if directory not in self.sources:
            root = QtWidgets.QTreeWidgetItem([directory.name]); root.setData(0, QtCore.Qt.UserRole, str(directory))
            root.setToolTip(0, str(directory))
            self.tree.addTopLevelItem(root); root.setExpanded(False)
            self.sources[directory] = root
        root = self.sources[directory]
        root.takeChildren()
        names = list(pipeline.REQUIRED)
        if self.inverse:
            names = [name for name in names if name != 'Heat_Flow_Basal.xlsx']
            names += ['TemperatureWellData.xlsx', 'VitriniteWellData.xlsx']
            if (directory / 'Heat_Flow_Basal.xlsx').is_file():
                names.append('Heat_Flow_Basal.xlsx')
        for name in names:
            if (directory / name).is_file():
                child = QtWidgets.QTreeWidgetItem([name]); child.setToolTip(0, str(directory/name)); root.addChild(child)
        self.active = directory
        self.tree.setCurrentItem(self.sources[directory])
        self.append_log(f'Loaded folder: {directory}')
        if self.tabs.currentWidget() is self.visual:
            self.tab_changed()

    def select_item(self, item, _column):
        if not self.running:
            while item.parent() is not None:
                item = item.parent()
            self.active = Path(item.data(0, QtCore.Qt.UserRole))
            self.tab_changed()

    def pick_folder(self):
        if self.picker_task or self.running:
            return
        self.pick_button.setEnabled(False)
        parent = f'x11:{int(self.window.winId()):x}'
        current = str(self.active or Path.home())
        self.picker_task = Task(lambda _emit, cancelled: asyncio.run(portal_directory(parent, 'Pick a folder with data', current, cancelled)))
        def done(path):
            self.picker_task = None; self.pick_button.setEnabled(not self.running)
            if path:
                self.add_directory(path)
        def failed(error):
            done(None); self.append_log(str(error))
        self.picker_task.signals.result.connect(done); self.picker_task.signals.error.connect(failed)
        QtCore.QThreadPool.globalInstance().start(self.picker_task)

    def tab_changed(self, *_):
        if self.tabs.currentWidget() is self.visual and self.active:
            result = self.results.get(self.active)
            self.viewer.load(result or self.active, preferred='Temperature' if result else None)

    def set_running(self, running):
        self.running = running
        self.table.setEnabled(not running); self.tree.setEnabled(not running)
        self.pick_button.setEnabled(not running); self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)

    def start(self):
        if self.running:
            return
        try:
            if self.active is None:
                raise ValueError('Pick a folder with data first.')
            options = self.engine.validate_options(self.values())
        except Exception as error:
            self.append_log(str(error)); return
        source = self.active
        # The sketches contain one folder picker; result runs live next to inputs.
        output = source.parent / 'results' / source.name
        if not os.access(source.parent, os.W_OK):
            output = Path(QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.DocumentsLocation)) / 'Basin Solver Results' / source.name
        self.runner = self.engine.Pipeline(self.append_log)
        self.set_running(True)
        self.append_log(f'Starting {source.name}')
        self.task = Task(lambda emit, _: self._run(source, output, options, emit))
        def complete(result):
            self.results[source] = result; self.set_running(False); self.task = None
            self.append_log(f'Result: {result}')
            if self.tabs.currentWidget() is self.visual:
                self.tab_changed()
        def failed(error):
            self.set_running(False); self.task = None
            self.append_log('Stopped' if isinstance(error, pipeline.Cancelled) else f'ERROR: {error}')
        self.task.signals.result.connect(complete); self.task.signals.error.connect(failed)
        self.task.signals.log.connect(self.append_log)
        QtCore.QThreadPool.globalInstance().start(self.task)

    def _run(self, source, output, options, emit):
        self.runner.emit = emit
        return self.runner.run(source, output, options)

    def stop(self):
        if self.running and self.stop_button.isEnabled():
            self.append_log('Stopping…'); self.stop_button.setEnabled(False); self.runner.cancel()


class StandalonePage(QtWidgets.QWidget):
    def __init__(self, window):
        super().__init__()
        self.window = window; self.task = None
        layout = QtWidgets.QVBoxLayout(self); layout.setContentsMargins(18, 28, 18, 18); layout.setSpacing(24)
        frame = QtWidgets.QFrame(); frame.setObjectName('panel')
        inner = QtWidgets.QVBoxLayout(frame); inner.setContentsMargins(12, 12, 12, 12)
        self.viewer = ViewerHost(); inner.addWidget(self.viewer)
        layout.addWidget(frame, 1)
        self.button = QtWidgets.QPushButton('Choose a folder\nwith data'); self.button.setFixedSize(170, 52)
        self.button.clicked.connect(self.pick_folder); layout.addWidget(self.button, 0, QtCore.Qt.AlignLeft)

    def pick_folder(self):
        if self.task:
            return
        self.button.setEnabled(False)
        parent = f'x11:{int(self.window.winId()):x}'
        self.task = Task(lambda _emit, cancelled: asyncio.run(portal_directory(parent, 'Choose a folder with data', str(Path.home()), cancelled)))
        def done(path):
            self.task = None; self.button.setEnabled(True)
            if path:
                self.viewer.load(path)
        def failed(error):
            done(None); self.viewer.clear(); self.viewer.layout.addWidget(QtWidgets.QLabel(str(error)))
        self.task.signals.result.connect(done); self.task.signals.error.connect(failed)
        QtCore.QThreadPool.globalInstance().start(self.task)


class TitleBar(QtWidgets.QFrame):
    def __init__(self, window):
        super().__init__(); self.setObjectName('title'); self.window = window; self.drag = None
        self.setFixedHeight(43)
        layout = QtWidgets.QHBoxLayout(self); layout.setContentsMargins(12, 0, 10, 0)
        layout.addSpacing(90); layout.addStretch()
        brand = QtWidgets.QLabel('Basin Solver'); brand.setObjectName('brand'); layout.addWidget(brand)
        layout.addStretch()
        for label, action in [('−',window.showMinimized),('□',self.maximize),('×',window.request_close)]:
            button = QtWidgets.QPushButton(label); button.setObjectName('windowControl'); button.setFixedSize(28, 30)
            button.clicked.connect(action); layout.addWidget(button)

    def maximize(self):
        self.window.showNormal() if self.window.isMaximized() else self.window.showMaximized()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.drag = event.globalPos() - self.window.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if self.drag is not None and event.buttons() & QtCore.Qt.LeftButton:
            self.window.move(event.globalPos() - self.drag)

    def mouseReleaseEvent(self, event):
        self.drag = None

    def mouseDoubleClickEvent(self, event):
        self.maximize()


class InvisibleGrip(QtWidgets.QSizeGrip):
    def paintEvent(self, event):
        pass


class Window(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName('window')
        self.setWindowTitle('Basin Solver'); self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.resize(600, 600); self.setMinimumSize(540, 540)
        self.pages = {}
        outside = QtWidgets.QVBoxLayout(self); outside.setContentsMargins(0, 0, 0, 0)
        shell = QtWidgets.QFrame(); shell.setObjectName('shell'); outside.addWidget(shell)
        inside = QtWidgets.QVBoxLayout(shell); inside.setContentsMargins(1, 1, 1, 1); inside.setSpacing(0)
        inside.addWidget(TitleBar(self))
        self.stack = QtWidgets.QStackedWidget(); inside.addWidget(self.stack, 1)
        self.home = QtWidgets.QWidget(); home_layout = QtWidgets.QVBoxLayout(self.home)
        home_layout.setContentsMargins(96, 63, 96, 55); home_layout.setSpacing(58)
        for text, mode in [('Forward solver','forward'),('Inverse solver','inverse'),('Visualize data','viewer')]:
            button = QtWidgets.QPushButton(text); button.setObjectName('homeAction'); button.setFixedHeight(72)
            button.clicked.connect(lambda _=False, mode=mode:self.open_page(mode)); home_layout.addWidget(button)
        home_layout.addStretch(); self.stack.addWidget(self.home)
        self.grip = InvisibleGrip(self); self.grip.setFixedSize(14, 14)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QtCore.QTimer.singleShot(0, self.apply_masks)
        if hasattr(self, 'grip'):
            self.grip.move(self.width()-16, self.height()-16)
            self.grip.raise_()

    def apply_masks(self):
        radius = 0 if self.isMaximized() else 20
        outer = QtGui.QPainterPath()
        outer.addRoundedRect(QtCore.QRectF(self.rect()), radius, radius)
        self.setMask(QtGui.QRegion(outer.toFillPolygon().toPolygon()))
        inner = QtGui.QPainterPath()
        inner.addRoundedRect(QtCore.QRectF(self.rect()).adjusted(2,2,-2,-2), max(0,radius-2), max(0,radius-2))
        origin = self.stack.mapTo(self, QtCore.QPoint(0,0))
        self.stack.setMask(QtGui.QRegion(inner.toFillPolygon().toPolygon()).translated(-origin.x(),-origin.y()))

    def open_page(self, mode):
        if mode not in self.pages:
            self.pages[mode] = StandalonePage(self) if mode == 'viewer' else ModePage(self, inverse=mode=='inverse')
            self.stack.addWidget(self.pages[mode])
        self.setMinimumSize(850, 680) if mode != 'viewer' else self.setMinimumSize(680, 680)
        self.resize(1100, 840) if mode != 'viewer' else self.resize(800, 900)
        self.stack.setCurrentWidget(self.pages[mode])

    def show_home(self):
        self.stack.setCurrentWidget(self.home); self.setMinimumSize(540, 540); self.resize(600, 600)

    def request_close(self):
        page = self.stack.currentWidget()
        if page is self.home:
            self.close()
        elif isinstance(page, ModePage) and page.running:
            page.stop()
            QtCore.QTimer.singleShot(100, self.request_close)
        else:
            picker = page.picker_task if isinstance(page, ModePage) else page.task
            if picker:
                picker.cancelled.set()
            self.show_home()

    def closeEvent(self, event):
        if any(isinstance(page, ModePage) and page.running for page in self.pages.values()):
            event.ignore()
            for page in self.pages.values():
                if isinstance(page, ModePage):
                    page.stop()
            QtCore.QTimer.singleShot(100, self.close)
            return
        for page in self.pages.values():
            picker = page.picker_task if isinstance(page, ModePage) else page.task
            if picker:
                picker.cancelled.set()
            viewer = page.viewer
            if viewer.task:
                viewer.task.cancelled.set()
        event.accept()


def run_gui():
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName('Basin Solver'); app.setDesktopFileName('basin-solver')
    app.setStyle('Fusion'); app.setStyleSheet(STYLE)
    palette=app.palette(); palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor('#292929'))
    palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor('#dddddd')); app.setPalette(palette)
    window = Window(); window.show()
    if '--ui-test' in sys.argv:
        import ui_check
        ui_check.attach(window, sys.argv[sys.argv.index('--ui-test')+1:])
    sys.exit(app.exec_())

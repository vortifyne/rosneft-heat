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
from profile_view import ProfileButton

BG = '#111111'
pg.setConfigOptions(background=BG, foreground='#dddddd', useOpenGL=False)
MODE_NAMES = {'forward': 'Forward solver', 'inverse': 'Inverse solver', 'viewer': 'Visualization'}
FORWARD = pipeline.EDITABLE
INVERSE = inverse_pipeline.INVERSE_PARAMETERS
STYLE = '''
QWidget#window { background: transparent; }
QWidget { background: #111111; color: #dddddd; font-family: "DejaVu Sans"; font-size: 12px; }
QFrame#shell { border: 2px solid #cccccc; border-radius: 20px; }
QFrame#shell[expanded="true"] { border-radius: 0; }
QToolTip { background: #242424; color: #dddddd; border: 1px solid #bbbbbb; padding: 5px; }
QFrame#title { border: none; border-bottom: 1px solid #bbbbbb; background: transparent; }
QLabel#brand { font-size: 21px; background: transparent; }
QPushButton { border: 1px solid #cccccc; border-radius: 9px; padding: 8px 16px; background: #111111; }
QPushButton:hover { background: #242424; }
QPushButton:checked { background: #262626; }
QPushButton:disabled { color: #696969; border-color: #555555; }
QPushButton#windowControl { border: none; padding: 0; font-size: 21px; border-radius: 0; background: transparent; }
QPushButton#homeAction { border-radius: 16px; font-size: 23px; }
QSplitter::handle { background: #525252; }
QSplitter::handle:hover { background: #aaaaaa; }
QFrame#panel { border: 1px solid #bbbbbb; border-radius: 18px; }

QLabel#treeTitle, QLabel#logTitle { font-size: 21px; }
QTreeWidget { border: none; outline: none; background: #111111; }
QTreeWidget::item { padding: 6px 2px; }
QTreeWidget::item:selected { background: #292929; }
QTabWidget::pane { border: none; }
QTabBar::tab { border: 1px solid #cccccc; border-radius: 6px; padding: 5px 18px; margin: 4px 3px 4px 0; }
QTabBar::tab:selected { background: #242424; }
QTableWidget { background: #111111; border: none; gridline-color: #aaaaaa; selection-background-color: #292929; }
QTableWidget QLineEdit { border: none; background: #111111; padding: 5px 8px; }
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


class FolderChoice(QtWidgets.QComboBox):
    def __init__(self, placeholder):
        super().__init__()
        self.setPlaceholderText(placeholder)
        self.setSizeAdjustPolicy(self.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(15)
        self.setFixedHeight(30)
        self.setEnabled(False)
        self.currentIndexChanged.connect(lambda i: self.setToolTip(self.itemData(i) or ''))

    def add_folder(self, directory, label=None, *, select=True):
        path = str(directory)
        index = self.findData(path)
        if index < 0:
            label = label or directory.name
            label_role = QtCore.Qt.UserRole + 1
            other = self.findData(label, label_role)
            text = label
            if other >= 0:
                self.setItemText(other, f'{label} — {Path(self.itemData(other)).parent}')
                text = f'{label} — {directory.parent}'
            index = self.count()
            self.addItem(text, path)
            self.setItemData(index, label, label_role)
            self.setItemData(index, path, QtCore.Qt.ToolTipRole)
        self.setEnabled(True)
        if select or self.currentIndex() < 0:
            self.setCurrentIndex(index)


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
        self.latest_results = {}

        self.runner = None
        self.running = False
        self.task = None
        self.picker_task = None
        self.defaults = self.engine.read_defaults()
        body = QtWidgets.QHBoxLayout(self)
        body.setContentsMargins(0, 0, 0, 0); body.setSpacing(0)
        columns = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        body.addWidget(columns)
        columns.setChildrenCollapsible(False)
        columns.setHandleWidth(6)
        sidebar = QtWidgets.QFrame()
        sidebar.setMinimumWidth(100)
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(12, 10, 8, 10)
        title = QtWidgets.QLabel('Tree'); title.setObjectName('treeTitle')
        sidebar_layout.addWidget(title)
        self.tree = FolderTree(); self.tree.setHeaderHidden(True); self.tree.setExpandsOnDoubleClick(False)
        self.tree.setIndentation(18)
        self.tree.currentItemChanged.connect(self.select_item)
        sidebar_layout.addWidget(self.tree)
        columns.addWidget(sidebar)
        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        right.setChildrenCollapsible(False)
        right.setHandleWidth(6)
        columns.addWidget(right)
        columns.setStretchFactor(1, 1)
        columns.setSizes([220, 880])
        self.tabs = QtWidgets.QTabWidget()
        self.main = QtWidgets.QWidget(); self.visual = QtWidgets.QWidget()
        self.tabs.addTab(self.main, 'main'); self.tabs.addTab(self.visual, 'visualize')
        right.addWidget(self.tabs)
        main_layout = QtWidgets.QVBoxLayout(self.main)
        main_layout.setContentsMargins(18, 4, 18, 14); main_layout.setSpacing(8)
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
        main_layout.addWidget(self.table, 1)
        fields = QtWidgets.QFormLayout()
        self.source_choice = FolderChoice('Load a data folder first')
        self.source_choice.currentIndexChanged.connect(self.source_changed)
        fields.addRow('Input data', self.source_choice)
        self.run_name = QtWidgets.QLineEdit()
        self.run_name.setPlaceholderText('Automatic name')
        fields.addRow('Result name suffix', self.run_name)
        main_layout.addLayout(fields)
        self.pick_button = QtWidgets.QPushButton('Pick a folder with data')
        self.start_button = QtWidgets.QPushButton('Start')
        self.stop_button = QtWidgets.QPushButton('Stop'); self.stop_button.setEnabled(False)
        self.performance = ProfileButton()
        actions = QtWidgets.QHBoxLayout()
        main_layout.addLayout(actions)
        for button in (self.pick_button, self.start_button, self.stop_button, self.performance):
            actions.addWidget(button)
        self.start_button.setEnabled(False)
        self.pick_button.clicked.connect(self.pick_folder)
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        visual_layout = QtWidgets.QVBoxLayout(self.visual)
        visual_layout.setContentsMargins(18, 4, 18, 14)
        self.result_choice = FolderChoice('Load input data or complete a calculation')
        result_form = QtWidgets.QFormLayout()
        self.result_performance = ProfileButton()
        selection = QtWidgets.QHBoxLayout()
        selection.addWidget(self.result_choice, 1)
        selection.addWidget(self.result_performance)
        result_form.addRow('Viewing', selection)
        self.result_choice.currentIndexChanged.connect(lambda _: self.result_performance.select(self.result_choice.currentData()))
        visual_layout.addLayout(result_form)
        self.result_choice.currentIndexChanged.connect(self.tab_changed)
        self.tabs.currentChanged.connect(self.tab_changed)
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
        log_frame = QtWidgets.QFrame(); log_frame.setObjectName('logPanel')
        log_layout = QtWidgets.QHBoxLayout(log_frame); log_layout.setContentsMargins(10, 8, 8, 8)
        log_title = QtWidgets.QLabel('LOG:'); log_title.setObjectName('logTitle'); log_title.setAlignment(QtCore.Qt.AlignTop)
        log_layout.addWidget(log_title)
        self.log = QtWidgets.QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumBlockCount(5000)
        self.log.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap); log_layout.addWidget(self.log, 1)
        right.addWidget(log_frame)
        right.setStretchFactor(0, 1)
        right.setSizes([650, 145])

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
        if directory in self.sources:
            anchor = self.pick_button
            QtWidgets.QToolTip.showText(anchor.mapToGlobal(anchor.rect().topLeft()),
                                        f'Folder already in the tree: {directory.name}', anchor)
            return
        if not directory.is_dir():
            self.append_log(f'Folder does not exist: {directory}'); return
        root = QtWidgets.QTreeWidgetItem([directory.name])
        root.setData(0, QtCore.Qt.UserRole, str(directory))
        root.setToolTip(0, str(directory))
        self.tree.addTopLevelItem(root)
        root.setExpanded(False)
        self.sources[directory] = root
        names = list(pipeline.REQUIRED)
        if self.inverse:
            names = [name for name in names if name != 'Heat_Flow_Basal.xlsx']
            names += ['TemperatureWellData.xlsx', 'VitriniteWellData.xlsx']
            if (directory / 'Heat_Flow_Basal.xlsx').is_file():
                names.append('Heat_Flow_Basal.xlsx')
        for name in names:
            if (directory / name).is_file():
                child = QtWidgets.QTreeWidgetItem([name]); child.setToolTip(0, str(directory/name)); root.addChild(child)
        self.source_choice.add_folder(directory)
        self.result_choice.add_folder(directory, f'Input preview · {directory.name}', select=False)
        self.append_log(f'Loaded folder: {directory}')

    @property
    def active(self):
        path = self.source_choice.currentData()
        return Path(path) if path else None

    def source_changed(self):
        self.performance.select(self.latest_results.get(self.active))
        self.tree.setCurrentItem(self.sources.get(self.active))
        self.start_button.setEnabled(self.active is not None and not self.running)

    def select_item(self, item, _previous):
        if item is not None and not self.running:
            while item.parent() is not None:
                item = item.parent()
            self.source_choice.setCurrentIndex(self.source_choice.findData(item.data(0, QtCore.Qt.UserRole)))

    def add_result(self, result):
        self.latest_results[self.active] = result
        self.performance.select(result)
        self.result_choice.add_folder(result)
        self.window.record_result(result, 'Inverse' if self.inverse else 'Forward')

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
        directory = self.result_choice.currentData()
        if self.tabs.currentWidget() is self.visual and directory:
            preferred = None if Path(directory) in self.sources else 'Temperature'
            self.viewer.load(directory, preferred=preferred)

    def set_running(self, running):
        self.running = running
        self.table.setEnabled(not running); self.tree.setEnabled(not running)
        self.pick_button.setEnabled(not running); self.start_button.setEnabled(not running)
        self.source_choice.setEnabled(not running and self.source_choice.count() > 0)
        self.run_name.setEnabled(not running)
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
        run_name = self.run_name.text().strip() or None
        output = source.parent / 'results' / source.name
        if pipeline.root_dir() in output.parents or not os.access(source.parent, os.W_OK):
            output = Path(QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.DocumentsLocation)) / 'Basin Solver Results' / source.name
        self.runner = self.engine.Pipeline(self.append_log)
        self.set_running(True)
        self.append_log(f'Starting {source.name}')
        self.task = Task(lambda emit, _: self._run(source, output, options, run_name, emit))
        def complete(result):
            self.set_running(False); self.task = None
            self.add_result(result)
            self.append_log(f'Result: {result}')
        def failed(error):
            self.set_running(False); self.task = None
            self.append_log('Stopped' if isinstance(error, pipeline.Cancelled) else f'ERROR: {error}')
        self.task.signals.result.connect(complete); self.task.signals.error.connect(failed)
        self.task.signals.log.connect(self.append_log)
        QtCore.QThreadPool.globalInstance().start(self.task)

    def _run(self, source, output, options, run_name, emit):
        self.runner.emit = emit
        return self.runner.run(source, output, options, run_name)

    def stop(self):
        if self.running and self.stop_button.isEnabled():
            self.append_log('Stopping…'); self.stop_button.setEnabled(False); self.runner.cancel()


class StandalonePage(QtWidgets.QWidget):
    def __init__(self, window):
        super().__init__()
        self.window = window; self.task = None
        layout = QtWidgets.QVBoxLayout(self); layout.setContentsMargins(18, 28, 18, 18); layout.setSpacing(24)
        self.choice = FolderChoice('Choose a folder or complete a calculation')
        form = QtWidgets.QFormLayout()
        self.performance = ProfileButton()
        selection = QtWidgets.QHBoxLayout()
        selection.addWidget(self.choice, 1)
        selection.addWidget(self.performance)
        form.addRow('Viewing', selection)
        self.choice.currentIndexChanged.connect(lambda _: self.performance.select(self.choice.currentData()))
        layout.addLayout(form)
        self.choice.currentIndexChanged.connect(self.load_selected)
        frame = QtWidgets.QFrame(); frame.setObjectName('panel')
        inner = QtWidgets.QVBoxLayout(frame); inner.setContentsMargins(12, 12, 12, 12)
        self.viewer = ViewerHost(); inner.addWidget(self.viewer)
        layout.addWidget(frame, 1)
        self.button = QtWidgets.QPushButton('Choose a folder\nwith data'); self.button.setFixedSize(170, 52)
        self.button.clicked.connect(self.pick_folder); layout.addWidget(self.button, 0, QtCore.Qt.AlignLeft)

    def load_selected(self):
        if self.isVisible() and self.choice.currentData():
            self.viewer.load(self.choice.currentData())

    def pick_folder(self):
        if self.task:
            return
        self.button.setEnabled(False)
        parent = f'x11:{int(self.window.winId()):x}'
        self.task = Task(lambda _emit, cancelled: asyncio.run(portal_directory(parent, 'Choose a folder with data', str(Path.home()), cancelled)))
        def done(path):
            self.task = None; self.button.setEnabled(True)
            if path:
                self.choice.add_folder(Path(path).resolve())
                self.load_selected()
        def failed(error):
            done(None); self.viewer.clear(); self.viewer.layout.addWidget(QtWidgets.QLabel(str(error)))
        self.task.signals.result.connect(done); self.task.signals.error.connect(failed)
        QtCore.QThreadPool.globalInstance().start(self.task)


class TitleBar(QtWidgets.QFrame):
    def __init__(self, window):
        super().__init__(); self.setObjectName('title'); self.window = window; self.drag = None
        self.setFixedHeight(43)
        layout = QtWidgets.QHBoxLayout(self); layout.setContentsMargins(12, 0, 10, 0)
        self.back = QtWidgets.QPushButton('←  Main menu')
        back_policy = self.back.sizePolicy()
        back_policy.setRetainSizeWhenHidden(True)
        self.back.setSizePolicy(back_policy)
        self.back.clicked.connect(window.show_home)
        layout.addWidget(self.back); self.back.hide(); layout.addStretch()
        self.brand = QtWidgets.QLabel('Basin Solver'); self.brand.setObjectName('brand'); layout.addWidget(self.brand)
        layout.addStretch()
        for label, action in [('−',window.showMinimized),('□',self.maximize),('×',window.close)]:
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


class ResizeHandle(QtWidgets.QWidget):
    def __init__(self, window, edges, cursor):
        super().__init__(window)
        self.setStyleSheet("background: transparent;")
        self.edges = edges
        self.setCursor(cursor)
        self.origin = None

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            window = self.window()
            if not window.windowHandle().startSystemResize(self.edges):
                self.origin = event.globalPos()
                self.initial = window.geometry()

    def mouseMoveEvent(self, event):
        if self.origin is None or not event.buttons() & QtCore.Qt.LeftButton:
            return
        window = self.window()
        delta = event.globalPos() - self.origin
        rect = QtCore.QRect(self.initial)
        if self.edges & QtCore.Qt.LeftEdge:
            rect.setLeft(min(rect.left() + delta.x(), rect.right() + 1 - window.minimumWidth()))
        if self.edges & QtCore.Qt.RightEdge:
            rect.setWidth(max(window.minimumWidth(), rect.width() + delta.x()))
        if self.edges & QtCore.Qt.TopEdge:
            rect.setTop(min(rect.top() + delta.y(), rect.bottom() + 1 - window.minimumHeight()))
        if self.edges & QtCore.Qt.BottomEdge:
            rect.setHeight(max(window.minimumHeight(), rect.height() + delta.y()))
        window.setGeometry(rect)

    def mouseReleaseEvent(self, event):
        self.origin = None


class Window(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName('window')
        self.setWindowTitle('Basin Solver'); self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.resize(600, 600); self.setMinimumSize(540, 540)
        self.pages = {}
        self.results = {}
        outside = QtWidgets.QVBoxLayout(self); outside.setContentsMargins(0, 0, 0, 0)
        self.shell = QtWidgets.QFrame(); self.shell.setObjectName('shell'); outside.addWidget(self.shell)
        inside = QtWidgets.QVBoxLayout(self.shell); inside.setContentsMargins(1, 1, 1, 1); inside.setSpacing(0)
        self.title_bar = TitleBar(self)
        inside.addWidget(self.title_bar)
        self.stack = QtWidgets.QStackedWidget(); inside.addWidget(self.stack, 1)
        self.home = QtWidgets.QWidget(); home_layout = QtWidgets.QVBoxLayout(self.home)
        home_layout.setContentsMargins(96, 63, 96, 55); home_layout.setSpacing(58)
        for mode, text in MODE_NAMES.items():
            button = QtWidgets.QPushButton(text); button.setObjectName('homeAction'); button.setFixedHeight(72)
            button.clicked.connect(lambda _=False, mode=mode:self.open_page(mode)); home_layout.addWidget(button)
        home_layout.addStretch(); self.stack.addWidget(self.home)
        qt = QtCore.Qt
        self.handles = []
        for edges, cursor in (
                (qt.LeftEdge, qt.SizeHorCursor), (qt.RightEdge, qt.SizeHorCursor),
                (qt.TopEdge, qt.SizeVerCursor), (qt.BottomEdge, qt.SizeVerCursor),
                (qt.TopEdge | qt.LeftEdge, qt.SizeFDiagCursor),
                (qt.BottomEdge | qt.RightEdge, qt.SizeFDiagCursor),
                (qt.TopEdge | qt.RightEdge, qt.SizeBDiagCursor),
                (qt.BottomEdge | qt.LeftEdge, qt.SizeBDiagCursor)):
            self.handles.append(ResizeHandle(self, edges, cursor))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_frame()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.WindowStateChange:
            self.update_frame()

    def update_frame(self):
        expanded = self.isMaximized() or self.isFullScreen()
        if self.shell.property('expanded') != expanded:
            self.shell.setProperty('expanded', expanded)
            self.shell.style().unpolish(self.shell)
            self.shell.style().polish(self.shell)
            self.shell.update()
        qt = QtCore.Qt
        for handle in self.handles:
            handle.setVisible(not expanded)
            if expanded:
                continue
            edges = handle.edges
            horizontal = edges & (qt.LeftEdge | qt.RightEdge)
            vertical = edges & (qt.TopEdge | qt.BottomEdge)
            thickness = 16 if horizontal and vertical else 6
            width = thickness if horizontal else self.width()
            height = thickness if vertical else self.height()
            x = self.width() - width if edges & qt.RightEdge else 0
            y = self.height() - height if edges & qt.BottomEdge else 0
            handle.setGeometry(x, y, width, height)
            handle.raise_()
        QtCore.QTimer.singleShot(0, self.apply_masks)

    def apply_masks(self):
        if self.isMaximized() or self.isFullScreen():
            self.clearMask()
            self.stack.clearMask()
            return
        radius = 20
        outer = QtGui.QPainterPath()
        outer.addRoundedRect(QtCore.QRectF(self.rect()), radius, radius)
        self.setMask(QtGui.QRegion(outer.toFillPolygon().toPolygon()))
        inner = QtGui.QPainterPath()
        inner.addRoundedRect(QtCore.QRectF(self.rect()).adjusted(2,2,-2,-2), max(0,radius-2), max(0,radius-2))
        origin = self.stack.mapTo(self, QtCore.QPoint(0,0))
        self.stack.setMask(QtGui.QRegion(inner.toFillPolygon().toPolygon()).translated(-origin.x(),-origin.y()))

    def record_result(self, path, mode):
        label = f'{mode} · {path.name}'
        self.results[path] = label
        if 'viewer' in self.pages:
            self.pages['viewer'].choice.add_folder(path, label, select=False)

    def set_heading(self, text, back):
        self.title_bar.brand.setText(text)
        self.title_bar.back.setVisible(back)
        self.setWindowTitle(text)

    def open_page(self, mode):
        if mode not in self.pages:
            self.pages[mode] = StandalonePage(self) if mode == 'viewer' else ModePage(self, inverse=mode=='inverse')
            self.stack.addWidget(self.pages[mode])
            if mode == 'viewer':
                for path, label in self.results.items():
                    self.pages[mode].choice.add_folder(path, label)
        self.setMinimumSize(850, 680) if mode != 'viewer' else self.setMinimumSize(680, 680)
        if not (self.isMaximized() or self.isFullScreen()):
            self.resize(1100, 840) if mode != 'viewer' else self.resize(800, 900)
        self.set_heading(MODE_NAMES[mode], True)
        self.stack.setCurrentWidget(self.pages[mode])
        if mode == 'viewer':
            self.pages[mode].load_selected()

    def show_home(self):
        self.set_heading('Basin Solver', False)
        self.stack.setCurrentWidget(self.home); self.setMinimumSize(540, 540)
        if not (self.isMaximized() or self.isFullScreen()):
            self.resize(600, 600)

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

"""Performance dialog for a selected result directory."""
from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from run_profile import STAGES, read_profile


def duration(seconds):
    if seconds < 1:
        return f'{seconds * 1000:.0f} ms'
    if seconds < 60:
        return f'{seconds:.2f} s'
    return f'{int(seconds // 60)} min {seconds % 60:04.1f} s'


def chart(layout, rows, total):
    grid = QtWidgets.QGridLayout()
    grid.setColumnStretch(1, 1)
    layout.addLayout(grid)
    for i, (label, seconds) in enumerate(sorted(rows, key=lambda item: -item[1])):
        percent = 100 * seconds / total if total else 0
        grid.addWidget(QtWidgets.QLabel(label), i, 0)
        bar = QtWidgets.QProgressBar()
        bar.setTextVisible(False)
        bar.setRange(0, 1000)
        bar.setValue(round(percent * 10))
        bar.setFixedHeight(9)
        bar.setStyleSheet('QProgressBar {background:#292929;border:0;border-radius:4px;}'
                          'QProgressBar::chunk {background:#cccccc;border-radius:4px;}')
        grid.addWidget(bar, i, 1)
        grid.addWidget(QtWidgets.QLabel(f'{duration(seconds)}   ·   {percent:.1f}%'), i, 2,
                       QtCore.Qt.AlignRight)


def description(layout, text, heading=False):
    label = QtWidgets.QLabel(text)
    label.setTextFormat(QtCore.Qt.PlainText)
    label.setWordWrap(True)
    if heading:
        font = label.font()
        font.setBold(True)
        font.setPointSize(13)
        label.setFont(font)
    layout.addWidget(label)


def solver_note(report):
    counts = report['counts']
    details = report['solver_s']
    facts = []
    for key, label in (('timesteps', 'time steps'), ('nonlinear_iterations', 'nonlinear iterations'),
                       ('linear_iterations', 'linear iterations'), ('forward_evaluations', 'forward evaluations'),
                       ('parameters', 'fitted parameters'), ('observations', 'observations')):
        if key in counts:
            facts.append(f"{counts[key]} {label}")
    note = ', '.join(facts) + '.' if facts else ''
    if details:
        largest = max(details, key=details.get)
        hint = {
            'Linear solve': 'Compare linear solver and preconditioner settings on the same input and check convergence.',
            'Assembly': 'Each nonlinear iteration rebuilds system terms. For exploratory runs, compare a coarser mesh and check the accuracy.',
            'Output': 'The native solver spent this time constructing and writing fields and snapshots.',
            'Input, inversion and parameter output': 'Each objective evaluation runs a forward model. Check the fit and termination reason before changing the evaluation limit.',
        }.get(largest)
        if hint:
            note += ' ' + hint
    return note.strip()


class ProfileButton(QtWidgets.QPushButton):
    def __init__(self):
        super().__init__('Performance')
        self.directory = None
        self.setEnabled(False)
        self.clicked.connect(self.show_report)

    def select(self, directory):
        self.directory = Path(directory) if directory else None
        available = self.directory is not None and (self.directory / 'performance.json').is_file()
        self.setEnabled(available)
        self.setToolTip(str(self.directory) if available else 'No performance report for this selection')

    def show_report(self):
        try:
            report = read_profile(self.directory)
        except (OSError, ValueError) as error:
            QtWidgets.QMessageBox.warning(self, 'Performance report', str(error))
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle('Run performance')
        dialog.resize(660, 540)
        outer = QtWidgets.QVBoxLayout(dialog)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        outer.addWidget(scroll)
        body = QtWidgets.QWidget()
        scroll.setWidget(body)
        layout = QtWidgets.QVBoxLayout(body)
        layout.setSpacing(12)
        description(layout, self.directory.name, heading=True)
        stages = report['stages_s']
        total = report['total_s']
        largest = max(stages, key=stages.get)
        name, reason = STAGES.get(largest, (largest, ''))
        description(layout, f'Total {duration(total)}   ·   Longest stage: {name} '
                    f'({100 * stages[largest] / total:.1f}%).', heading=True)
        description(layout, reason)
        chart(layout, [(STAGES.get(key, (key,))[0], value) for key, value in stages.items()], total)
        counts = report['counts']
        if counts.get('exported_workbooks') and 'exported_bytes' in counts:
            size = counts['exported_bytes'] / (1024 * 1024)
            description(layout, f"Export wrote {counts['exported_workbooks']} Excel workbooks "
                        f"({size:.1f} MiB on disk). Verification read them back.")
        if largest in ('prepare', 'validate', 'export', 'check'):
            description(layout, 'If the files are on a network drive, compare this stage on local storage '
                        'using the same inputs. Timings include process startup.')
        if report['solver_s']:
            description(layout, 'Inside the solver · share of solver time', heading=True)
            chart(layout, report['solver_s'].items(), stages['solve'])
        note = solver_note(report)
        if note:
            description(layout, note)
        layout.addStretch()
        close = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        close.rejected.connect(dialog.reject)
        outer.addWidget(close)
        dialog.exec_()

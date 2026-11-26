"""Build a complete AppDir from freshly compiled solver and frozen Python GUI."""
from pathlib import Path
import json
import os
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / 'build/AppDir-unified'
FREEZE = ROOT / 'build/freeze-unified'
if APP.exists():
    shutil.rmtree(APP)
APP.mkdir(parents=True)
subprocess.run(['python3', '-m', 'PyInstaller', '--noconfirm', '--clean', '--onedir',
    '--name', 'basin_solver', '--distpath', str(FREEZE), '--workpath', str(ROOT / 'build/pyinstaller-unified'),
    '--specpath', str(ROOT / 'build'), '--paths', str(ROOT / 'tools'),
    '--paths', str(ROOT / 'desktop'), '--hidden-import', 'matplotlib.backends.backend_qt5agg',
    '--exclude-module', 'tkinter', '--exclude-module', 'PySide2', '--exclude-module', 'PySide6', '--exclude-module', 'PyQt6',
    '--collect-data', 'matplotlib', str(ROOT / 'desktop/main.py')], check=True)
shutil.copytree(FREEZE / 'basin_solver', APP / 'usr/bin')
# Do not use an executable left in an old AppDir.
for name in ('basin_forward_solver', 'basin_inverse_solver'):
    shutil.copy2(ROOT / 'build/docker/release' / name, APP / 'usr/bin' / name)
for name in ('AppRun', 'run_solver', 'basin-solver.desktop', 'basin-solver.svg'):
    shutil.copy2(ROOT / 'packaging/linux' / name, APP / name)
(APP / 'run_inverse_solver').write_text((APP / 'run_solver').read_text().replace('basin_forward_solver', 'basin_inverse_solver'))
(APP / 'run_inverse_solver').chmod(0o755)
(APP / '.DirIcon').symlink_to('basin-solver.svg')
(APP / 'AppRun').chmod(0o755)
(APP / 'run_solver').chmod(0o755)
libs = APP / 'usr/lib'
libs.mkdir(parents=True, exist_ok=True)
# An intentionally empty, owned plugin directory isolates hwloc from the host.
# Linux CPU/memory discovery is built into libhwloc; GPU plugins are unnecessary.
(libs / 'hwloc').mkdir()
# MPI and PMIx load modules dynamically, so ldd on the solver alone is insufficient.
plugins = []
for name, pattern in [('openmpi', '/usr/lib/x86_64-linux-gnu/openmpi/lib/openmpi*'),
                      ('pmix', '/usr/lib/x86_64-linux-gnu/pmix*/lib/pmix*')]:
    import glob
    for directory in glob.glob(pattern):
        if Path(directory).is_dir():
            target = libs / name
            shutil.copytree(directory, target, dirs_exist_ok=True)
            plugins.extend(target.rglob('*.so'))
for source, destination in [('/usr/share/openmpi', 'usr/share/openmpi'),
                             ('/etc/openmpi', 'usr/etc/openmpi')]:
    if Path(source).is_dir():
        shutil.copytree(source, APP / destination, dirs_exist_ok=True)
if Path('/usr/bin/orted').exists():
    shutil.copy2('/usr/bin/orted', APP / 'usr/bin/orted.real')
    launcher = APP / 'usr/bin/orted'
    subprocess.run(['cc', '-std=c11', '-D_POSIX_C_SOURCE=200809L', '-O2',
                    '-Wall', '-Wextra', '-Werror', '-static',
                    str(ROOT / 'packaging/linux/orted_launcher.c'),
                    '-o', str(launcher)], check=True)
    headers = subprocess.check_output(['readelf', '-l', str(launcher)], text=True)
    if 'INTERP' in headers:
        raise RuntimeError('orted trampoline must not use the host ELF interpreter')
queue = list((APP / 'usr/bin').rglob('*')) + plugins
seen = set()
while queue:
    path = queue.pop()
    if not path.is_file() or path in seen:
        continue
    seen.add(path)
    result = subprocess.run(['ldd', str(path)], capture_output=True, text=True)
    if 'not found' in result.stdout:
        raise RuntimeError(f'Missing dependencies for {path}: {result.stdout}')
    for match in re.findall(r'(?:=>\s+)?(/[^\s()]+)', result.stdout):
        dependency = Path(match)
        target = libs / dependency.name
        if dependency.is_file() and not target.exists():
            shutil.copy2(dependency.resolve(), target)
            queue.append(target)
licenses = APP / 'licenses'
licenses.mkdir()
from importlib.metadata import distribution
# dbus-next is pure Python; keep its MIT notice with the runtime licenses.
for package in ('dbus-next', 'PyQt5', 'PyQt5-Qt5', 'pyqtgraph'):
    metadata = distribution(package)
    for entry in metadata.files or ():
        if entry.name.upper().startswith(('LICENSE', 'COPYING')):
            shutil.copy2(metadata.locate_file(entry), licenses / (package + '-' + entry.name))
for copyright_file in Path('/usr/share/doc').glob('*/copyright'):
    shutil.copy2(copyright_file, licenses / (copyright_file.parent.name + '.copyright'))
subprocess.run(['python3', '-m', 'pip', 'freeze'], stdout=(APP / 'python-dependencies.txt').open('w'), check=True)
shutil.copytree(ROOT / 'docs/user', APP / 'docs')
(APP / 'build-info.json').write_text(json.dumps({
    'revision': subprocess.check_output(['git', '-c', f'safe.directory={ROOT}', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    'dirty': bool(subprocess.check_output(['git', '-c', f'safe.directory={ROOT}', 'status', '--porcelain'], cwd=ROOT, text=True)),
    'format': 'basin-solver-unified', 'platform': 'Linux x86_64 / Ubuntu 24.04 build'}, indent=2))
for path in APP.rglob('*'):
    if not path.is_symlink():
        path.chmod(path.stat().st_mode | 0o444 | (0o111 if path.is_dir() else 0))
print(APP)

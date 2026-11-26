"""Probe the shipped inverse MPI runtime against a stalled display and host GL plugins.

Only the executable in a copy of AppRun is changed, retaining its exact bundled
loader and environment. Stop immediately after MPI initializes; do not optimize.
"""
import argparse
import json
import os
from pathlib import Path
import selectors
import signal
import socket
import subprocess
import tempfile
import threading
import time

p = argparse.ArgumentParser()
p.add_argument('image', type=Path)
p.add_argument('input', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
a.image, a.input, a.output = a.image.resolve(), a.input.resolve(), a.output.resolve()
a.output.mkdir(parents=True, exist_ok=True)
connections = []
stop = threading.Event()
socket_path = Path('/tmp/.X11-unix/X197')
socket_path.parent.mkdir(exist_ok=True)
server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(str(socket_path))
server.listen()
server.settimeout(.2)


def accept():
    while not stop.is_set():
        try:
            connections.append(server.accept()[0])
        except socket.timeout:
            pass
        except OSError:
            return


thread = threading.Thread(target=accept, daemon=True)
thread.start()
try:
    with tempfile.TemporaryDirectory(prefix='inverse-mpi-probe-') as directory:
        root = Path(directory)
        subprocess.run([str(a.image), '--appimage-extract'], cwd=root, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=30)
        app = root / 'squashfs-root'
        probe = app / 'ProbeRun'
        probe.write_text((app / 'AppRun').read_text().replace('usr/bin/basin_solver', 'usr/bin/basin_inverse_solver'))
        probe.chmod(0o755)
        env = os.environ.copy()
        env.update(DISPLAY=':197', HWLOC_GL='1', HWLOC_PLUGINS_VERBOSE='1')
        process = subprocess.Popen([str(probe), str(a.input), str(a.output / 'unused'), '--maxiter', '1', '--maxfev', '1'],
                                   env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        data = b''
        ready = False
        child_commands = []
        try:
            os.set_blocking(process.stdout.fileno(), False)
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                deadline = time.monotonic() + 20
                while time.monotonic() < deadline and process.poll() is None:
                    if selector.select(.1):
                        block = os.read(process.stdout.fileno(), 65536)
                        data += block
                        if b'PETSc/MPI initialized' in data:
                            ready = True
                            children = Path(f'/proc/{process.pid}/task/{process.pid}/children').read_text().split()
                            for child in children:
                                try:
                                    child_commands.append(Path(f'/proc/{child}/cmdline').read_bytes().decode().replace('\0', ' '))
                                except FileNotFoundError:
                                    pass
                            break
        finally:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(3)
            process.stdout.close()
        (a.output / 'mpi.log').write_bytes(data)
        env['LD_LIBRARY_PATH'] = str(app / 'usr/lib')
        # Direct trampoline invocation needs the data prefix normally supplied by AppRun.
        env.update(OPAL_PREFIX=str(app / 'usr'), OPAL_DATADIR=str(app / 'usr/share'),
                   OPAL_PKGDATADIR=str(app / 'usr/share/openmpi'))
        daemon = subprocess.run([str(app / 'usr/bin/orted'), '--help'], env=env, capture_output=True, text=True, timeout=10)
        (a.output / 'orted.log').write_text(daemon.stdout + daemon.stderr)
        report = {'mpi_initialized': ready, 'stalled_x_connections': len(connections),
                  'mpi_children': child_commands, 'orted_trampoline_exit': daemon.returncode,
                  'orted_help': (daemon.stdout + daemon.stderr).strip(),
                  'orted_trampoline_loaded': 'Usage:' in daemon.stdout + daemon.stderr}
        (a.output / 'probe.json').write_text(json.dumps(report, indent=2))
        assert ready and not connections and not any('orted' in command for command in child_commands), report
        # ORTE's help path exits 1; reaching its usage text proves the bundled
        # loader/runtime ran successfully (a loader error exits before this).
        assert report['orted_trampoline_loaded'] and 'GLIBC_PRIVATE' not in daemon.stderr, report
        assert 'undefined symbol' not in daemon.stderr, report
        print(json.dumps(report), flush=True)
finally:
    stop.set()
    server.close()
    for connection in connections:
        connection.close()
    thread.join(1)
    socket_path.unlink(missing_ok=True)

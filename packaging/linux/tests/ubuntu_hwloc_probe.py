"""Reproduce MPI init hanging on a stalled X display through host hwloc GL.

Run inside a disposable Ubuntu container. It creates a private X socket, not a
real display, then runs the actual AppImage/XLSX pipeline against organizer data.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import threading

parser = argparse.ArgumentParser()
parser.add_argument('image', type=Path)
parser.add_argument('data', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--expect', choices=('blocked', 'success'), required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
socket_root = Path('/tmp/.X11-unix')
socket_root.mkdir(exist_ok=True)
socket_path = socket_root / 'X0'
connections = []
stopped = threading.Event()
server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(str(socket_path))
server.listen(32)
server.settimeout(.2)


def accept_connections():
    while not stopped.is_set():
        try:
            connection, _ = server.accept()
            # Deliberately never send an X11 setup reply. An optional GL probe
            # must not block a CPU-only heat solver on this unrelated display.
            connections.append(connection)
        except socket.timeout:
            continue
        except OSError:
            return


thread = threading.Thread(target=accept_connections, daemon=True)
thread.start()
env = os.environ.copy()
env.update(DISPLAY=':0', HWLOC_GL='1', HWLOC_PLUGINS_VERBOSE='1', HWLOC_COMPONENTS_VERBOSE='1')
log_path = args.output / 'probe.log'
try:
    with log_path.open('w') as log:
        result = subprocess.run([str(args.image), '--appimage-extract-and-run', '--run',
            str(args.data), str(args.output / 'results'), '--dx', '100'], env=env,
            stdout=log, stderr=subprocess.STDOUT, timeout=100)
    text = log_path.read_text()
    report = {'returncode': result.returncode, 'x_connections': len(connections),
              'mpi_initialized': 'PETSc/MPI initialized' in text,
              'mpi_timeout': 'initialization timed out' in text,
              'success': 'SUCCESS: Calculation completed' in text}
    (args.output / 'probe.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if args.expect == 'blocked':
        assert report['x_connections'] > 0 and report['mpi_timeout'], report
        assert not report['mpi_initialized'] and result.returncode != 0, report
    else:
        assert report['x_connections'] == 0 and report['mpi_initialized'], report
        assert report['success'] and result.returncode == 0, report
finally:
    stopped.set()
    server.close()
    for connection in connections:
        connection.close()
    socket_path.unlink(missing_ok=True)
    thread.join(1)

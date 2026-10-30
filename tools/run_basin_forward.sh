#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: tools/run_basin_forward.sh <prepared-input-dir> <output-dir> [basin_forward_solver-options...]" >&2
    exit 1
fi

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
input_dir=$(realpath -e -- "$1")
mkdir -p -- "$2"
output_dir=$(realpath -e -- "$2")
shift 2
binary=${BASIN_FORWARD_BINARY:-$repo_dir/build/docker/release/basin_forward_solver}
if [[ ! -x "$binary" ]]; then
    echo "Executable not found: $binary. Build basin_forward_solver first." >&2
    exit 1
fi
binary=$(realpath -e -- "$binary")

exec docker run --rm --network none \
    --user "$(id -u):$(id -g)" \
    -v "$binary:/app/basin_forward_solver:ro" \
    -v "$input_dir:/input:ro" -v "$output_dir:/output" \
    -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
    -e MKL_NUM_THREADS=1 -e BLIS_NUM_THREADS=1 \
    "${DOCKER_IMAGE:-heat-env}" /app/basin_forward_solver /input /output "$@"

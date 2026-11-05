#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: tools/run_basin_inverse.sh <prepared-input-dir> <output-dir> [basin_inverse_solver-options...]" >&2
    exit 1
fi

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
input_dir=$(realpath -e -- "$1")
mkdir -p -- "$2"
output_dir=$(realpath -e -- "$2")
shift 2
binary=${BASIN_INVERSE_BINARY:-$repo_dir/build/docker/release/basin_inverse_solver}
if [[ ! -x "$binary" ]]; then
    echo "Executable not found: $binary. Build basin_inverse_solver first." >&2
    exit 1
fi
binary=$(realpath -e -- "$binary")

options=("$@")
parameter_mount=()
for ((i=0; i<${#options[@]}; ++i)); do
    if [[ ${options[i]} == --initial-parameters ]]; then
        parameters=$(realpath -e -- "${options[i+1]}")
        parameter_mount=(-v "$parameters:/initial_parameters.csv:ro")
        options[i+1]=/initial_parameters.csv
        break
    fi
done

exec docker run --rm --init --network none \
    --user "$(id -u):$(id -g)" \
    -v "$binary:/app/basin_inverse_solver:ro" \
    -v "$input_dir:/input:ro" -v "$output_dir:/output" \
    "${parameter_mount[@]}" \
    -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
    -e MKL_NUM_THREADS=1 -e BLIS_NUM_THREADS=1 \
    "${DOCKER_IMAGE:-heat-env}" /app/basin_inverse_solver /input /output "${options[@]}"

#!/usr/bin/env bash
set -euo pipefail

solver=forward
if [[ ${1:-} == --inverse ]]; then
    solver=inverse
    shift
fi

if [[ $# -lt 2 ]]; then
    echo "Usage: tools/run_basin_from_xlsx.sh [--inverse] <spreadsheet-dir> <output-dir> [solver-options...]" >&2
    exit 1
fi

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
data_dir=$(realpath -e -- "$1")
mkdir -p -- "$2"
output_dir=$(realpath -e -- "$2")
shift 2

UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/rosneft-heat-uv-cache} \
    uv run --frozen --project "$repo_dir" python "$repo_dir/tools/prepare_basin_input.py" \
    "$data_dir" "$output_dir/input"
export_options=()
if [[ $solver == inverse ]]; then
    set -- --save-forward "$@"
    export_options=(--inverse)
fi
"$repo_dir/tools/run_basin_${solver}.sh" "$output_dir/input" "$output_dir" "$@"
UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/rosneft-heat-uv-cache} \
    uv run --frozen --project "$repo_dir" python "$repo_dir/tools/export_basin_results.py" \
    "$data_dir" "$output_dir" "${export_options[@]}"

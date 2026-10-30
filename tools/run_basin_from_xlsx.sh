#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: tools/run_basin_from_xlsx.sh <spreadsheet-dir> <output-dir> [basin_forward_solver-options...]" >&2
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
exec "$repo_dir/tools/run_basin_forward.sh" "$output_dir/input" "$output_dir" "$@"

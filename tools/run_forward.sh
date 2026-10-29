#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: tools/run_forward.sh <spreadsheet-dir> <output-dir> [heat_forward-options...]" >&2
    exit 1
fi

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
data_dir=$(realpath -e -- "$1")
mkdir -p -- "$2"
output_dir=$(realpath -e -- "$2")
shift 2

UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/rosneft-heat-uv-cache} \
    uv run --frozen --project "$repo_dir" python "$repo_dir/tools/prepare_forward_data.py" \
    "$data_dir" "$output_dir/input" --basin-data "$repo_dir/tools/basin_data.py"
exec "$repo_dir/tools/run_heat_forward.sh" "$output_dir/input" "$output_dir" "$@"

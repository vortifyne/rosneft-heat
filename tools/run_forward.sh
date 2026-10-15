#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: tools/run_forward.sh <data-dir> <output-dir> [cell-size] [dt-ma] [solver-options...]" >&2
    exit 1
fi

data_dir=$1
output_dir=$2
shift 2
cell_size=50
dt_ma=0.25
if [[ $# -gt 0 && $1 != --* ]]; then
    cell_size=$1
    shift
fi
if [[ $# -gt 0 && $1 != --* ]]; then
    dt_ma=$1
    shift
fi
input_dir="$output_dir/input"

UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/rosneft-heat-uv-cache} \
    uv run python tools/prepare_forward_data.py "$data_dir" "$input_dir" "$cell_size"
tools/run_heat_forward.sh "$input_dir" "$output_dir" --cell-size "$cell_size" --dt-ma "$dt_ma" "$@"

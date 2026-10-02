#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
    echo "Usage: tools/run_forward.sh <data-dir> <output-dir> [cell-size] [dt-ma]" >&2
    exit 1
fi

data_dir=$1
output_dir=$2
cell_size=${3:-50}
dt_ma=${4:-0.25}
input_dir="$output_dir/input"

UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/rosneft-heat-uv-cache} \
    uv run python tools/prepare_forward_data.py "$data_dir" "$input_dir" "$cell_size"
build/release/heat_forward "$input_dir" "$output_dir" --cell-size "$cell_size" --dt-ma "$dt_ma"

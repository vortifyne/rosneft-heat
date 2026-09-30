#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 6 ]]; then
    echo "Usage: tools/run_forward.sh <data-dir> <output-dir> [cell-size] [dt-ma] [save-every] [heat-forward]" >&2
    exit 1
fi

data_dir=$1
output_dir=$2
cell_size=${3:-50}
dt_ma=${4:-0.1}
save_every=${5:-10}
solver=${6:-build/release/heat_forward}
input_dir="$output_dir/input"

uv run python tools/prepare_forward_data.py "$data_dir" "$input_dir" \
    --boundary-step "$cell_size"
"$solver" "$input_dir" "$output_dir" "$cell_size" "$dt_ma" "$save_every"

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 12 ]]; then
    echo "Usage: tools/run_forward.sh <data-dir> <output-dir> [cell-size] [dt-ma] [states-per-epoch] [heat-forward] [max-configurations] [comparison-points] [thin-layer-cell-fraction] [boundary-step] [mesh-kind] [estimate-condition]" >&2
    exit 1
fi

data_dir=$1
output_dir=$2
cell_size=${3:-50}
dt_ma=${4:-0.1}
states_per_epoch=${5:-5}
solver=${6:-build/release/heat_forward}
max_configurations=${7:-}
comparison_points=${8:-}
thin_layer_cell_fraction=${9:-}
boundary_step=${10:-$cell_size}
mesh_kind=${11:-layered}
estimate_condition=${12:-no}
input_dir="$output_dir/input"

prepare_arguments=("$data_dir" "$input_dir" --boundary-step "$boundary_step")
solver_arguments=("$input_dir" "$output_dir" --cell-size "$cell_size" --dt-ma "$dt_ma" --states-per-epoch "$states_per_epoch")
if [[ -n "$max_configurations" ]]; then
    prepare_arguments+=(--max-configurations "$max_configurations")
    solver_arguments+=(--max-configurations "$max_configurations")
fi
if [[ -n "$comparison_points" ]]; then
    solver_arguments+=(--comparison-points "$comparison_points")
fi
if [[ -n "$thin_layer_cell_fraction" ]]; then
    solver_arguments+=(--thin-layer-cell-fraction "$thin_layer_cell_fraction")
fi
if [[ "$mesh_kind" == "layered" ]]; then
    solver_arguments+=(--layered-mesh)
elif [[ "$mesh_kind" == "quad-dominant" ]]; then
    solver_arguments+=(--quad-dominant)
elif [[ "$mesh_kind" == "triangular" ]]; then
    solver_arguments+=(--triangular-mesh)
else
    echo "Unknown mesh kind: $mesh_kind" >&2
    exit 1
fi
if [[ "$estimate_condition" == "yes" ]]; then
    solver_arguments+=(--estimate-condition)
elif [[ "$estimate_condition" != "no" ]]; then
    echo "Unknown estimate-condition value: $estimate_condition" >&2
    exit 1
fi

UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/rosneft-heat-uv-cache} \
    uv run python tools/prepare_forward_data.py "${prepare_arguments[@]}"
"$solver" "${solver_arguments[@]}"

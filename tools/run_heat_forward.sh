#!/usr/bin/env bash
set -euo pipefail

# Run the Docker-built executable with its PETSc runtime dependencies.
# Input and output paths are relative to the repository or absolute paths inside it.
exec docker run --rm \
    --user "$(id -u):$(id -g)" \
    -v "$PWD:$PWD" -w "$PWD" \
    -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
    -e MKL_NUM_THREADS=1 -e BLIS_NUM_THREADS=1 \
    heat-env build/release/heat_forward "$@"

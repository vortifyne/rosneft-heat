#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_dir"
image=${HEAT_PACKAGE_IMAGE:-basin-solver-package}
if [[ ${HEAT_SKIP_IMAGE_BUILD:-0} != 1 ]]; then
    docker build -t "$image" -f packaging/linux/Dockerfile .
fi
product=BasinSolver
appdir=build/AppDir-unified
mkdir -p dist local/appimage-tools
# VCPKG_MANIFEST_INSTALL=OFF is only for an existing dependency tree.
# Set HEAT_OFFLINE_DEPS=1 to reuse it; fresh builds install the manifest normally.
offline=${HEAT_OFFLINE_DEPS:-0}
docker run --rm -v "$repo_dir:/workspace" -v vcpkg-cache:/var/cache/vcpkg \
    -e HEAT_OFFLINE_DEPS="$offline" "$image" bash -c '
    cmake --preset docker-release -DVCPKG_MANIFEST_INSTALL=$([[ $HEAT_OFFLINE_DEPS == 1 ]] && echo OFF || echo ON)
    cmake --build --preset docker-release --parallel "${HEAT_JOBS:-4}"
    python3 packaging/linux/bundle.py
    '
runtime="$repo_dir/local/appimage-tools/runtime-x86_64"
if [[ ! -f "$runtime" ]]; then
    curl -fL --retry 3 https://github.com/AppImage/type2-runtime/releases/download/continuous/runtime-x86_64 -o "$runtime"
fi
if [[ -f packaging/linux/runtime.sha256 ]]; then
    (cd local/appimage-tools && sha256sum --check "$repo_dir/packaging/linux/runtime.sha256")
fi
# Standard type-2 ELF AppImage: the official runtime followed by a SquashFS payload.
package=$(mktemp "$repo_dir/dist/${product}.XXXXXX.AppImage")
cp "$runtime" "$package"
payload=$(mktemp "$repo_dir/dist/${product}.XXXXXX.squashfs")
trap 'rm -f -- "$payload" "$package"' EXIT
mksquashfs "$appdir" "$payload" -noappend -all-root -comp gzip -processors 2 -no-progress
cat "$payload" >> "$package"
chmod 755 "$package"
mv -f "$package" "dist/${product}-x86_64.AppImage"
(cd dist && sha256sum "${product}-x86_64.AppImage" > "${product}-x86_64.AppImage.sha256")

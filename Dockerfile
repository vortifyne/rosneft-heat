FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV VCPKG_ROOT=/opt/vcpkg
ENV PATH="${VCPKG_ROOT}:${PATH}"

ENV VCPKG_DEFAULT_BINARY_CACHE=/var/cache/vcpkg

RUN apt-get update && apt-get install -y \
  build-essential g++ clang ninja-build \
  git curl zip unzip tar pkg-config gfortran \
  liblapack-dev libblas-dev autoconf autoconf-archive \
  automake libtool ca-certificates \
  && rm -rf /var/lib/apt/lists/*

RUN CMAKE_VERSION=3.31.5 && \
  curl -sSL https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz | \
  tar --strip-components=1 -xz -C /usr/local

RUN git clone --depth 1 --branch 2026.06.24 https://github.com/microsoft/vcpkg.git /opt/vcpkg && \
  /opt/vcpkg/bootstrap-vcpkg.sh -disableMetrics && \
  mkdir -p /var/cache/vcpkg && \
  chmod -R 777 /opt/vcpkg /var/cache/vcpkg

WORKDIR /workspace

CMD ["/bin/bash"]

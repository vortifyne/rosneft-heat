FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV VCPKG_ROOT=/opt/vcpkg
ENV PATH="${VCPKG_ROOT}:${PATH}"

RUN apt-get update && apt-get install -y \
    build-essential g++ clang cmake ninja-build \
    git curl zip unzip tar pkg-config gfortran \
    liblapack-dev libblas-dev autoconf autoconf-archive \
    automake libtool \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/microsoft/vcpkg.git /opt/vcpkg && \
    /opt/vcpkg/bootstrap-vcpkg.sh -disableMetrics

WORKDIR /workspace

CMD ["/bin/bash"]

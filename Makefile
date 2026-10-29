.DEFAULT_GOAL := help

.PHONY: help docker-image \
        docker-debug docker-release docker-test docker-bench \
        local-debug local-release local-test local-bench clean clean-deps

DOCKER_IMAGE ?= heat-env
JOBS ?= 4
MODEL ?= OFF
BENCH_REVISION = $(shell git rev-parse HEAD)
BENCH_DIRTY = $(shell git diff --quiet HEAD -- && echo false || echo true)
BENCH_ENV = HEAT_BENCHMARK_REVISION=$(BENCH_REVISION) HEAT_BENCHMARK_DIRTY=$(BENCH_DIRTY)
BENCH_ARGS ?=

DOCKER_RUN = docker run --rm \
	--user $(shell id -u):$(shell id -g) \
	-v "$(CURDIR):$(CURDIR)" \
	-v vcpkg-cache:/var/cache/vcpkg \
	-w "$(CURDIR)" \
	-e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
	-e MKL_NUM_THREADS=1 -e BLIS_NUM_THREADS=1 \
	$(DOCKER_IMAGE)

help:
	@echo "make docker-image                     Build the Docker environment"
	@echo "make {docker,local}-{debug,release}    Build basin_forward_solver"
	@echo "make {docker,local}-test [MODEL=ON]    Build and run tests"
	@echo "make {docker,local}-bench              Build and run benchmarks"
	@echo "make clean                            Remove build outputs"
	@echo "make clean-deps                       Remove installed dependencies"

# Configure and build with one preset; tests and benchmarks have separate builds.
define configure_build
cmake --preset $(1) -DHEAT_BUILD_MODEL=$(MODEL) && cmake --build --preset $(1) --parallel $(JOBS) $(if $(2),--target $(2))
endef

docker-image:
	docker build -t $(DOCKER_IMAGE) .

docker-debug docker-release:
	$(DOCKER_RUN) bash -c '$(call configure_build,$@)'

local-debug local-release:
	$(call configure_build,$(patsubst local-%,%,$@))

docker-test:
	$(DOCKER_RUN) bash -c '$(call configure_build,docker-test) && ctest --preset docker-test'

local-test:
	$(call configure_build,test)
	ctest --preset test

docker-bench:
	$(DOCKER_RUN) bash -c '$(call configure_build,docker-bench,benchmarks) && $(BENCH_ENV) build/docker/bench/benchmarks $(BENCH_ARGS)'

local-bench:
	$(call configure_build,bench,benchmarks)
	$(BENCH_ENV) build/local/bench/benchmarks $(BENCH_ARGS)

clean:
	rm -rf build/ compile_commands.json

clean-deps:
	rm -rf vcpkg_installed/

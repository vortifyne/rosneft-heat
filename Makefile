.DEFAULT_GOAL := help

.PHONY: help docker-image \
        docker-debug docker-release docker-test docker-bench \
        local-debug local-release local-test local-bench clean clean-deps

DOCKER_IMAGE ?= heat-env
JOBS ?= 4
MODEL ?= OFF
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
	@echo "make {docker,local}-{debug,release,test,bench} [MODEL=ON] [JOBS=4]"
	@echo "make docker-image | clean | clean-deps"

docker-image:
	docker build -t $(DOCKER_IMAGE) .

docker-debug docker-release:
	$(DOCKER_RUN) bash -c 'cmake --preset $@ -DHEAT_BUILD_MODEL=$(MODEL) && cmake --build --preset $@ --parallel $(JOBS)'

local-debug:
	cmake --preset debug -DHEAT_BUILD_MODEL=$(MODEL)
	cmake --build --preset debug --parallel $(JOBS)

local-release:
	cmake --preset release -DHEAT_BUILD_MODEL=$(MODEL)
	cmake --build --preset release --parallel $(JOBS)

docker-test:
	$(DOCKER_RUN) bash -c 'cmake --preset docker-test -DHEAT_BUILD_MODEL=$(MODEL) && cmake --build --preset docker-test --parallel $(JOBS) && ctest --preset docker-test'

local-test:
	cmake --preset test -DHEAT_BUILD_MODEL=$(MODEL)
	cmake --build --preset test --parallel $(JOBS)
	ctest --preset test

docker-bench:
	$(DOCKER_RUN) bash -c 'cmake --preset docker-bench -DHEAT_BUILD_MODEL=$(MODEL) && cmake --build --preset docker-bench --parallel $(JOBS) --target benchmarks && build/docker/bench/benchmarks $(BENCH_ARGS)'

local-bench:
	cmake --preset bench -DHEAT_BUILD_MODEL=$(MODEL)
	cmake --build --preset bench --parallel $(JOBS) --target benchmarks
	build/local/bench/benchmarks $(BENCH_ARGS)

clean:
	rm -rf build/ compile_commands.json

clean-deps:
	rm -rf vcpkg_installed/

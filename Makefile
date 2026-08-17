.PHONY: all help \
        docker-image docker-debug docker-release docker-run docker-bench docker-test \
        local-debug local-release local-run local-bench local-test clean

DOCKER_IMAGE = heat-env

DOCKER_RUN = docker run --rm \
	--user $(shell id -u):$(shell id -g) \
	-v $(shell pwd):$(shell pwd) \
	-v vcpkg-cache:/var/cache/vcpkg \
	-w $(shell pwd) \
	$(DOCKER_IMAGE)

# Docker (single runnable and benchmarkable)
docker-image:
	docker build -t $(DOCKER_IMAGE) .

docker-debug:
	$(DOCKER_RUN) bash -c "cmake --preset debug -DVCPKG_INSTALLED_DIR=$(shell pwd)/vcpkg_installed && cmake --build --preset debug"

docker-release:
	$(DOCKER_RUN) bash -c "cmake --preset release -DVCPKG_INSTALLED_DIR=$(shell pwd)/vcpkg_installed && cmake --build --preset release"

docker-run:
	$(DOCKER_RUN) ./build/release/heat_solver

docker-test:
	$(DOCKER_RUN) bash -c "ctest --test-dir build/release --output-on-failure"


# Local (resolve problems & testing)
local-debug:
	cmake --preset debug -DVCPKG_INSTALLED_DIR=$(shell pwd)/vcpkg_installed
	cmake --build --preset debug

local-release:
	cmake --preset release -DVCPKG_INSTALLED_DIR=$(shell pwd)/vcpkg_installed
	cmake --build --preset release

local-run:
	./build/release/heat_solver

local-test:
	ctest --test-dir build/release --output-on-failure

# General
clean:
	rm -rf build/ vcpkg_installed/ compile_commands.json

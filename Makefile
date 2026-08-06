.PHONY: docker-build build-debug build-release run clean test

docker-build:
	docker build -t heat-env .

build-debug:
	docker run --rm -v $(shell pwd):/workspace heat-env bash -c "cmake --preset debug && cmake --build --preset debug"

build-release:
	docker run --rm -v $(shell pwd):/workspace heat-env bash -c "cmake --preset release && cmake --build --preset release"

run:
	docker run --rm -v $(shell pwd):/workspace heat-env ./build/release/heat_solver

run-debug:
	docker run --rm -v $(shell pwd):/workspace heat-env ./build/debug/heat_solver

clean:
	rm -rf build/

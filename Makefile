.PHONY: docker-build build-debug build-release run run-debug test clean

docker-build:
	docker build -t heat-env .

build-debug:
	docker run --rm -v $(shell pwd):$(shell pwd) -w $(shell pwd) heat-env bash -c "cmake --preset debug && cmake --build --preset debug"

build-release:
	docker run --rm -v $(shell pwd):$(shell pwd) -w $(shell pwd) heat-env bash -c "cmake --preset release && cmake --build --preset release"

run:
	docker run --rm -v $(shell pwd):$(shell pwd) -w $(shell pwd) heat-env ./build/release/heat_solver

run-debug:
	docker run --rm -v $(shell pwd):$(shell pwd) -w $(shell pwd) heat-env ./build/debug/heat_solver

test:
	docker run --rm -v $(shell pwd):$(shell pwd) -w $(shell pwd) heat-env bash -c "ctest --test-dir build/debug --output-on-failure"

clean:
	rm -rf build/

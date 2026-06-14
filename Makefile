BUILD_DIR := build
CUDA_COMPILER := nvcc

.PHONY: configure build-dev build clean-config clean test

build-dev:
	cmake --build $(BUILD_DIR) --config Debug --clean-first -j --verbose

build:
	cmake --build $(BUILD_DIR) --config RelWithDebInfo --clean-first -j

configure:
	cmake -S . -B $(BUILD_DIR) \
		-G "Ninja Multi-Config" \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
		-DCMAKE_CUDA_COMPILER=$(CUDA_COMPILER)

test:
	ctest --test-dir $(BUILD_DIR) --output-on-failure

clean:
	cmake --build $(BUILD_DIR) --target clean

clean-config:
	rm -rf $(BUILD_DIR)

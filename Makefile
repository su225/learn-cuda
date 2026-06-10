BUILD_DIR := build
CUDA_COMPILER := nvcc

.PHONY: configure build clean test

dev:
	cmake -S . -B $(BUILD_DIR) -G Ninja \
		-DCMAKE_BUILD_TYPE=Debug \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
		-DCMAKE_CUDA_COMPILER=nvcc
	cmake --build $(BUILD_DIR) -j

configure:
	cmake -S . -B $(BUILD_DIR) \
		-G Ninja \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
		-DCMAKE_CUDA_COMPILER=$(CUDA_COMPILER)

build:
	cmake --build $(BUILD_DIR) -j

test:
	ctest --test-dir $(BUILD_DIR) --output-on-failure

clean:
	rm -rf $(BUILD_DIR)

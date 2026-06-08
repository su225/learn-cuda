BUILD_DIR := build

.PHONY: configure build clean run

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
		-DCMAKE_CUDA_COMPILER=nvcc

build:
	cmake --build $(BUILD_DIR) -j

run:
	ctest --test-dir $(BUILD_DIR) --output-on-failure

clean:
	rm -rf $(BUILD_DIR)

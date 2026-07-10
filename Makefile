BUILD_DIR := build
CUDA_COMPILER := nvcc
CUDA_COMPUTE_SANITIZER := compute-sanitizer

KERNELS := tiled tiled_v2 naive cublas
RACECHECK_TARGETS := $(addprefix racecheck-, $(KERNELS))
MEMCHECK_TARGETS := $(addprefix memcheck-, $(KERNELS))

.PHONY: build-dev
build-dev:
	cmake --build $(BUILD_DIR) --config Debug --clean-first -j --verbose

.PHONY: build
build:
	cmake --build $(BUILD_DIR) --config RelWithDebInfo --clean-first -j

.PHONY: configure
configure:
	cmake -S . -B $(BUILD_DIR) \
		-G "Ninja Multi-Config" \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
		-DCMAKE_CUDA_COMPILER=$(CUDA_COMPILER)

.PHONY: racecheck
racecheck: $(RACECHECK_TARGETS)

.PHONY: racecheck-%
racecheck-%:
	$(CUDA_COMPUTE_SANITIZER) --tool racecheck \
	    $(BUILD_DIR)/bin/RelWithDebInfo/gpu_runner \
		--M 128 --N 128 --K 128 \
		--kernel $*

.PHONY: memcheck
memcheck: $(MEMCHECK_TARGETS)

.PHONY: memcheck-%
memcheck-%:
	$(CUDA_COMPUTE_SANITIZER) --tool memcheck \
	    $(BUILD_DIR)/bin/RelWithDebInfo/gpu_runner \
		--M 128 --N 128 --K 128 \
		--kernel $*

.PHONY: test
test:
	ctest --test-dir $(BUILD_DIR) --output-on-failure

.PHONY: clean
clean:
	cmake --build $(BUILD_DIR) --target clean

.PHONY: clean-config
clean-config:
	rm -rf $(BUILD_DIR)

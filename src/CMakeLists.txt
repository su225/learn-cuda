add_library(gpu_kernels
  gemm/naive.cu
  gemm/tiled.cu
  gemm/cublas.cu
)

target_include_directories(gpu_kernels
  PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(gpu_kernels
  PUBLIC CUDA::cudart
)

# GPU runner for benchmarking and running individual
# kernels are binaries

add_executable(gpu_runner
    runner/main.cu
)

target_link_libraries(gpu_runner
    PRIVATE
    gpu_kernels
    CUDA::cublas
    CUDA::cublasLt
)

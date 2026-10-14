function(enable_cuda_debug target)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>
  )
endfunction()

add_library(gpu_kernels
  gemm/naive.cu
  gemm/tiled.cu
  gemm/tiled_2.cu
  gemm/tiled_vectorized_2.cu
  gemm/tiled_async_2.cu
  gemm/tiled_async_swizzle_b.cu
  gemm/tiled_async_swizzle.cu
  gemm/tiled_vectorized_3.cu
  gemm/cublas.cu
  common.cpp
)

target_include_directories(gpu_kernels
  PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(gpu_kernels
  PUBLIC
    CUDA::cudart
    cute_headers
)

# GPU runner for benchmarking and running individual
# kernels are binaries

add_executable(gpu_runner
  runner/main.cu
)

target_link_libraries(gpu_runner PRIVATE
    gpu_kernels
    CUDA::cublas
    CUDA::cublasLt
)

enable_cuda_debug(gpu_kernels)
enable_cuda_debug(gpu_runner)

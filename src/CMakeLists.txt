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

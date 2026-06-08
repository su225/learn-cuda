add_library(gemm
  gemm/naive.cu
  gemm/tiled.cu
  gemm/cublas.cu
)

target_include_directories(gemm
  PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(gemm
  PUBLIC CUDA::cudart
)

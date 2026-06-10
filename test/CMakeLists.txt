add_executable(gpu_tests
    main.cpp
    gemm/test_naive_gemm.cu
    src/test_utils.cu
)

target_include_directories(gpu_tests
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIRECTORY}/include
)

target_link_libraries(gpu_tests
    PRIVATE
    gpu_kernels
    GTest::gtest
    GTest::gtest_main
    CUDA::cublas
    CUDA::cublasLt
)

enable_testing()
gtest_discover_tests(gpu_tests)

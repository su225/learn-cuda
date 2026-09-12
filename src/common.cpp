#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cassert>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#include "common.h"

const char* cublasGetErrorString(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
        default:
            return "Unknown cuBLAS status";
    }
}

std::vector<float> random_matrix(int size, float scale, int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-scale, scale);

    std::vector<float> v(size);
    for (auto& x : v)
        x = dist(gen);
    return v;
}

void print_matrix(
    std::vector<float> matrix,
    int rows,
    int cols,
    int maxPrintRows,
    int maxPrintCols
) {
    assert(matrix.size() == rows * cols);
    assert(rows >= maxPrintRows && cols >= maxPrintCols);
    for (int i = 0; i < maxPrintRows; i++) {
        for (int j = 0; j < maxPrintCols; j++) {
            std::cout << std::fixed
                << std::setprecision(4) << std::setw(8) << std::setfill(' ')
                << matrix[i*cols + j] << ' ';
        }
        std::cout << std::endl;
    }
}

template <int BM, int BN, int BK, int TM, int TN>
void validate_gemm_config() {
    static_assert(BM > 0 && BN > 0 && BK > 0, "BM, BN, and BK must be positive");
    static_assert(TM > 0 && TN > 0, "TM and TN must be positive");
    static_assert(BM % TM == 0, "BM must be divisible by TM");
    static_assert(BN % TN == 0, "BN must be divisible by TN");

    constexpr int block_x = BN / TN;
    constexpr int block_y = BM / TM;
    constexpr int threads = block_x * block_y;

    static_assert(threads <= 1024, "Too many threads per block");

    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    if (block_x > prop.maxThreadsDim[0]) {
        throw std::runtime_error("block_x exceeds device limit");
    }
    if (block_y > prop.maxThreadsDim[1]) {
        throw std::runtime_error("block_y exceeds device limit");
    }

    const std::size_t shared_bytes =
        (static_cast<std::size_t>(BM) * BK +
         static_cast<std::size_t>(BK) * BN) *
        sizeof(float);

    if (shared_bytes > prop.sharedMemPerBlock) {
        throw std::runtime_error("Requested shared memory exceeds device limit");
    }
    if (BM > prop.maxThreadsDim[1] * TM) {
        throw std::runtime_error("BM exceeds supported block mapping");
    }
    if (BN > prop.maxThreadsDim[0] * TN) {
        throw std::runtime_error("BN exceeds supported block mapping");
    }
}

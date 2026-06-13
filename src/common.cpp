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

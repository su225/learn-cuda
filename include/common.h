#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#define CUDA_CHECK(call)                                                   \
do {                                                                       \
	cudaError_t error = (call);                                              \
	if (error != cudaSuccess) {                                            \
		std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
				  << " code=" << static_cast<int>(error)                   \
				  << " reason=" << cudaGetErrorString(error) << std::endl; \
		std::exit(EXIT_FAILURE);                                           \
	}                                                                      \
} while (0);


#define CUBLAS_CHECK(call) \
do { \
	cublasStatus_t st = (call); \
	if (st != CUBLAS_STATUS_SUCCESS) { \
	    std::cerr << "cuBLASLt error at " << __FILE__ << ":" << __LINE__ \
				  << " code=" << static_cast<int>(st)                 \
				  << " reason=" << cublasGetErrorString(st) << std::endl; \
		std::exit(EXIT_FAILURE); \
	} \
} while (0);


#define INDEX(i,j,r,c) ((i*c + j))

const char* cublasGetErrorString(cublasStatus_t status);

std::vector<float> random_matrix(int size, float scale, int seed);

void print_matrix(std::vector<float> matrix,
                  int rows, int cols,
                  int maxPrintRows, int maxPrintCols);

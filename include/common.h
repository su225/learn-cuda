#pragma once

#include <iostream>
#include <cuda_runtime.h>
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
		std::cerr << "cuBLASLt error: " << st << std::endl; \
		std::exit(EXIT_FAILURE); \
	} \
} while (0);


#define INDEX(i,j,r,c) ((i*c + j))

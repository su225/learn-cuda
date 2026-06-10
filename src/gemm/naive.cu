#include <cstdio>
#include <vector>

#include <cuda_runtime.h>
#include "common.h"

__global__ void naive_gemm_kernel(float *A,
	            	  	          float *B,
					              float *C,
	                  	          int M,
								  int N,
	            	  	          int K) {
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	if (r < M && c < N) {
		float result = 0.0;
		for (int k = 0; k < K; k++) {
			result += A[r*K + k] * B[k*N + c];
		}
		C[r*N + c] = result;
	}
}

void naive_gemm(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int M,
    int N,
    int K
) {
    float *dA, *dB, *dC;

    CUDA_CHECK(cudaMalloc(&dA, A.size() * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&dB, B.size() * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&dC, C.size() * sizeof(float)));

	CUDA_CHECK(cudaMemcpy(dA, A.data(), A.size()*sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(dB, B.data(), B.size()*sizeof(float), cudaMemcpyHostToDevice));

	dim3 blockDim(32, 32);
	dim3 gridDim(
		(N + blockDim.x - 1) / blockDim.x, // columns
		(M + blockDim.y - 1) / blockDim.y  // rows
	);
	naive_gemm_kernel<<<gridDim, blockDim>>>(dA, dB, dC, M, N, K);
	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaMemcpy(C.data(), dC, C.size()*sizeof(float), cudaMemcpyDeviceToHost));
}

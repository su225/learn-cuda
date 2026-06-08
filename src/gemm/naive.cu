#include <cstdio>
#include <vector>

#include <cuda_runtime.h>
#include "common.h"

__global__ void naive_gemm(float *A,
	            	  	   float *B,
	                  	   float *C,
	                  	   int M,
	            	  	   int K,
	            	  	   int N) {
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

void launch_naive_gemm(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int M,
    int K,
    int N
) {
    size_t sizeA = M*K * sizeof(float);
    size_t sizeB = K*N * sizeof(float);
    size_t sizeC = M*N * sizeof(float);

    float *dA, *dB, *dC;

    CUDA_CHECK(cudaMalloc(&dA, sizeA));
	CUDA_CHECK(cudaMalloc(&dB, sizeB));
	CUDA_CHECK(cudaMalloc(&dC, sizeC));

	CUDA_CHECK(cudaMemcpy(dA, A.data(), M*K*sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(dB, B.data(), K*N*sizeof(float), cudaMemcpyHostToDevice));

	dim3 blockDim(32, 32);
	dim3 gridDim(
		(N + blockDim.x - 1) / blockDim.x,
		(M + blockDim.y - 1) / blockDim.y
	);
	naive_gemm<<<gridDim, blockDim>>>(dA, dB, dC, M, K, N);
	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaMemcpy(C.data(), dC, sizeC, cudaMemcpyDeviceToHost));
}

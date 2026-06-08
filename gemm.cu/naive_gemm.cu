#include <stdio.h>

#define CUDA_CHECK(call) \
{ \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason:%s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

#define INDEX(i,j,R,C) (i*C + j)

__global__ void naive_gemm(float *A,
	            	  	   float *B,
	                  	   float *C,
	                  	   int M,
	            	  	   int N,
	            	  	   int K,
	            	  	   float alpha,
	            	  	   float beta) {
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

int main() {
	float *h_A, *h_B, *h_C;
	float *d_A, *d_B, *d_C;
	int M = 1024;
	int N = 1024;
	int K = 1024;

	int elem_a = M * K;
	size_t size_a = sizeof(float) * elem_a;

	int elem_b = K * N;
	size_t size_b = sizeof(float) * elem_b;

	int elem_c = M * N;
	size_t size_c = sizeof(float) * elem_c;

	h_A = (float *)malloc(size_a);
	h_B = (float *)malloc(size_b);
	h_C = (float *)malloc(size_c);
	for (int i = 0; i < M; i++)
		for (int j = 0; j < K; j++)
			h_A[INDEX(i,j,M,K)] = 1.0*(i+j)/(M+K);

	for (int i = 0; i < K; i++)
		for (int j = 0; j < N; j++)
			h_B[INDEX(i,j,K,N)] = 1.0*(i+j)/(K+N);

	CUDA_CHECK(cudaMalloc(&d_A, size_a));
	CUDA_CHECK(cudaMalloc(&d_B, size_b));
	CUDA_CHECK(cudaMalloc(&d_C, size_c));

	CUDA_CHECK(cudaMemcpy(d_A, h_A, size_a, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_B, h_B, size_b, cudaMemcpyHostToDevice));

	dim3 blockDim(32, 32);
	dim3 gridDim(
		(N + blockDim.x - 1) / blockDim.x,
		(M + blockDim.y - 1) / blockDim.y
	);
	naive_gemm<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, 1.0, 1.0);
	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaMemcpy(h_C, d_C, size_c, cudaMemcpyDeviceToHost));

	for (int i = 0; i < 15; i++) {
		for (int j = 0; j < 15; j++) {
			printf("%.4f ", h_C[i*N+j]);
		}
		printf("\n");
	}

	CUDA_CHECK(cudaFree(d_A));
	CUDA_CHECK(cudaFree(d_B));
	CUDA_CHECK(cudaFree(d_C));
	free(h_A); free(h_B); free(h_C);

	return 0;
}
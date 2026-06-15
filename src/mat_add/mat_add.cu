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

__global__ void matrix_addition(float *A, float *B, float *C, int M, int N) {
	int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
	if (rowIdx < M && colIdx < N) {
		int elemIdx = rowIdx * N + colIdx;
		C[elemIdx] = A[elemIdx] + B[elemIdx];
	}
}

int main() {
	float *h_A, *h_B, *h_C;
	float *d_A, *d_B, *d_C;

	int rows = 1024 * 16;
	int cols = 1024 * 16;
	int totalElements = rows * cols;
	size_t size = sizeof(float) * totalElements;
	h_A = (float *)malloc(size);
	h_B = (float *)malloc(size);
	h_C = (float *)malloc(size);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			int idx = i * cols + j;
			h_A[idx] = i + j;
			h_B[idx] = (i + j) * 2.0;
		}
	}

	CUDA_CHECK(cudaMalloc(&d_A, size));
	CUDA_CHECK(cudaMalloc(&d_B, size));
	CUDA_CHECK(cudaMalloc(&d_C, size));

	CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

	// blockDim.x >= 32 because each warp has 32 threads. 32 threads loading
	// 4B single-precision floating point numbers matches the max possible
	// memory transaction coalescing which is 128B. So..When a warp is scheduled
	// and all threads in the warp issue read to adjacent memory locations, then
	// they can all be coalesced.
	//
	// if blockDim.x < 32, say 16, then the warp would have 2 rows of 16 elements
	// each which would mean two 64B coalesced transactions are needed. In fact, this
	// is a strided read (strided by the number of columns) and hence it breaks coalescing.
	dim3 blockDim(32, 8);
	dim3 gridDim(
		(cols + blockDim.x - 1) / blockDim.x,
		(rows + blockDim.y - 1) / blockDim.y
	);
	matrix_addition<<<gridDim, blockDim>>>(d_A, d_B, d_C, rows, cols);
	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

	// Verification/Smoke test to make sure that it is not off.
	for (int r = 0; r < 10; r++) {
		for (int c = 0; c < 10; c++) {
			int idx = r * cols + c;
			printf("%.4f ", h_C[idx]);
		}
		printf("\n");
	}

	CUDA_CHECK(cudaFree(d_A));
	CUDA_CHECK(cudaFree(d_B));
	CUDA_CHECK(cudaFree(d_C));
	free(h_A); free(h_B); free(h_C);

	return 0;
}

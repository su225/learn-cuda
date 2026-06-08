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

#define TILESZ 32

__global__ void tiled_gemm(float *A,
	            	  	   float *B,
	                  	   float *C,
	                  	   int M,
	            	  	   int N,
	            	  	   int K,
	            	  	   float alpha,
	            	  	   float beta) {
	__shared__ float tile_A[TILESZ][TILESZ]; // 32x32x4 = 4KiB
	__shared__ float tile_B[TILESZ][TILESZ]; // 32x32x4 = 4KiB

	int start_row = blockDim.y * blockIdx.y;
	int start_col = blockDim.x * blockIdx.x;

	int cur_warp_idx = threadIdx.y; 	  // y dimension represents the row. 1 warp = 1 row.
	int thread_idx_in_warp = threadIdx.x; // x dimension represents the col.

	float acc = 0.0;

	// Cooperatively load the 32x32 tile into shared memory.
	// In order to compute the tile that the block is responsible
	// for, we still need to start from the left and the top. Just
	// that we are loading 1 whole tile at a time.
	for (int t = 0; t < K; t += TILESZ) {
		// Since the block size of 32x32 corresponds to the shared memory
		// size of 32x32 with 32 threads forming a warp, each row of A's tile
		// and column of B's tile is assigned to a warp. Warp 0 (threads 0-31)
		// loads row 0 of A's tile and col 0 of B's tile.
		int tile_a_row = start_row + cur_warp_idx;
		int tile_a_col = t*TILESZ + thread_idx_in_warp;

		int tile_b_col = start_col + cur_warp_idx;
		int tile_b_row = t*TILESZ + thread_idx_in_warp;

		if ((tile_a_row < M && tile_a_col < K) &&
			(tile_b_row < K && tile_b_col < N)) {
			tile_A[threadIdx.y][threadIdx.x] = A[tile_a_row * K + tile_a_col];
			tile_B[threadIdx.y][threadIdx.x] = B[tile_b_row * N + tile_b_col];
		}
		__syncthreads();

		// Now perform matrix multiplication
		for (int k = 0; k < TILESZ; k++) {
			acc += tile_A[cur_warp_idx][k] * tile_B[k][cur_warp_idx];
		}
	}

	int cur_row = start_row + threadIdx.y;
	int cur_col = start_col + threadIdx.x;
	C[cur_row*N + cur_col] = acc;
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
			// h_A[i*M+j] = 1.0*(i+j)/(M+K);
			h_A[INDEX(i,j,M,K)] = 1.0*(i+j)/(M+K);

	for (int i = 0; i < K; i++)
		for (int j = 0; j < N; j++)
			// h_B[i*K+j] = 1.0*(i+j)/(K+N);
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
	tiled_gemm<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, 1.0, 1.0);
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
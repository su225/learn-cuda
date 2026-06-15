#include <cstdio>
#include <vector>

#include <cuda_runtime.h>
#include "common.h"

#define TILESZ 32

__global__ void tiled_gemm_kernel(float *A,
	            	  	          float *B,
					              float *C,
	                  	          int M,
								  int N,
	            	  	          int K) {
	__shared__ float tile_A[TILESZ][TILESZ]; // 32x32x4 = 4KiB
	__shared__ float tile_B[TILESZ][TILESZ]; // 32x32x4 = 4KiB

	int start_row = blockDim.y * blockIdx.y;
	int start_col = blockDim.x * blockIdx.x;

	float acc = 0.0;

	// Cooperatively load the 32x32 tile into shared memory.
	// In order to compute the tile that the block is responsible
	// for, we still need to start from the left and the top. Just
	// that we are loading 1 whole tile at a time.
	for (int tileIdx = 0; tileIdx < K/TILESZ; tileIdx++) {
		// Since the block size of 32x32 corresponds to the shared memory
		// size of 32x32 with 32 threads forming a warp, each row of A's tile
		// and column of B's tile is assigned to a warp. Warp 0 (threads 0-31)
		// loads row 0 of A's tile and col 0 of B's tile.
		int tile_a_row = start_row + threadIdx.y;
		int tile_a_col = tileIdx * TILESZ + threadIdx.x;
		int tile_b_row = tileIdx * TILESZ + threadIdx.y;
		int tile_b_col = start_col + threadIdx.x;

		if (tile_a_row < M && tile_a_col < K) {
		    tile_A[threadIdx.y][threadIdx.x] = A[tile_a_row * K + tile_a_col];
		} else {
		    tile_A[threadIdx.y][threadIdx.x] = 0.0f;
		}

		if (tile_b_row < K && tile_b_col < N) {
		    tile_B[threadIdx.y][threadIdx.x] = B[tile_b_row * N + tile_b_col];
		} else {
		    tile_B[threadIdx.y][threadIdx.x] = 0.0f;
		}
		__syncthreads();

		// Now perform matrix multiplication
		for (int k = 0; k < TILESZ; k++) {
			acc += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
		}
		// Why syncthreads again? This is to prevent race conditions. Imagine two
		// threads computing this kernel. One of them finishes computing partial
		// product through the tile while the other one is still reading from both
		// tile_A and tile_B. Without this __syncthreads(), the thread which finished
		// early will start loading and perhaps even rewriting the locations currently
		// read by the other, slower thread leading to incorrect results.
		__syncthreads();
	}

	int cur_row = start_row + threadIdx.y;
	int cur_col = start_col + threadIdx.x;
	if (cur_row < M && cur_col < N) {
	    C[cur_row*N + cur_col] = acc;
	}
}

void tiled_gemm(
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
        (M + blockDim.y - 1) / blockDim.y
    );
    tiled_gemm_kernel<<<gridDim, blockDim>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(C.data(), dC, C.size()*sizeof(float), cudaMemcpyDeviceToHost));
}

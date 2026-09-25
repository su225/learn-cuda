#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "common.h"

template <int BM, int BN, int BK, int TM, int TN>
__global__
void tiled_gemm_kernel_v2(float const* __restrict__ A,
                          float const* __restrict__ B,
                          float *C,
                          const int M,
                          const int N,
                          const int K) {
    static_assert(BM > 0 && BN > 0 && BK > 0, "Tile dimensions must be positive");
    static_assert(TM > 0 && TN > 0, "Thread tile dimensions must be positive");
    static_assert(BM % TM == 0, "BM must be divisible by TM");
    static_assert(BN % TN == 0, "BN must be divisible by TN");

    constexpr int THREADS_M = BM / TM;
    constexpr int THREADS_N = BN / TN;
    static_assert(THREADS_M * THREADS_N <= 1024, "Too many threads per block");

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    // The top-left (row,col) coordinates of the resulting submatrix
    // the current block/CTA is responsible for computing
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // The block BM*BN is further subdivided into thread-blocks which
    // is the area a single thread is responsible for. It is TM*TN.
    // This is the coordinate of the top-left (row,col) coordinates
    // of the area the current thread is responsible for.
    // We map tid -> ((blockRow,blockCol),(threadRow,threadCol))
    const int threadblock_row = block_row + ty * TM;
    const int threadblock_col = block_col + tx * TN;

    // Register accumulator
    float acc[TM][TN];

    #pragma unroll
    for (int m = 0; m < TM; m++) {
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            acc[m][n] = 0.0f;
        }
    }

    // Number of threads in the participating block.
    const int num_block_threads = blockDim.x * blockDim.y;

    // Iterate over K in BK-sized chunks.
    for (int k0 = 0; k0 < K; k0 += BK) {

        // Load tile A. All threads in the block load consecutive addresses. This
        // coalesces memory accesses assuming row-major storage. Each thread with
        // tid=T would load (T, T+B, T+2B,...) where B is the number of threads.
        for (int linear = tid; linear < BM * BK; linear += num_block_threads) {
            const int smem_row = linear / BK;
            const int smem_col = linear % BK;
            const int global_row = block_row + smem_row;
            const int global_col = k0 + smem_col;
            if (global_row < M && global_col < K) {
                As[smem_row][smem_col] = A[global_row * K + global_col];
            } else {
                As[smem_row][smem_col] = 0.0f;
            }
        }

        for (int linear = tid; linear < BK * BN; linear += num_block_threads) {
            const int smem_row = linear / BN;
            const int smem_col = linear % BN;
            const int global_row = k0 + smem_row;
            const int global_col = block_col + smem_col;
            if (global_row < K && global_col < N) {
                Bs[smem_row][smem_col] = B[global_row * N + global_col];
            } else {
                Bs[smem_row][smem_col] = 0.0f;
            }
        }
        __syncthreads();

        const int valid_k = (BK < (K-k0)) ? BK : (K-k0);
        for (int k = 0; k < valid_k; ++k) {
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                const float a = As[ty*TM + m][k];

                #pragma unroll
                for (int n = 0; n < TN; ++n) {
                    const float b = Bs[k][tx*TN + n];
                    acc[m][n] += a * b;
                }
            }
        }
        __syncthreads();
    }

    // Store the register tile
    const int max_valid_m = TM < (M-threadblock_row) ? TM : (M-threadblock_row);
    const int max_valid_n = TN < (N-threadblock_col) ? TN : (N-threadblock_col);
    # pragma unroll
    for (int m = 0; m < max_valid_m; m++) {
        const int global_row = threadblock_row + m;
        #pragma unroll
        for (int n = 0; n < max_valid_n; n++) {
            const int global_col = threadblock_col + n;
            C[global_row * N + global_col] = acc[m][n];
        }
    }
}

void tiled_gemm_v2(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int M, int N, int K
) {
    float *dA, *dB, *dC;

    CUDA_CHECK(cudaMalloc(&dA, A.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, B.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, C.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dA, A.data(), A.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), B.size()*sizeof(float), cudaMemcpyHostToDevice));

    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int TM = 8;
    constexpr int TN = 8;

    dim3 block(BN/TN, BM/TM); // 16x16 threads (256 threads per block)
    dim3 grid(
        (N + BN - 1)/BN,
        (M + BM - 1)/BM
    );

    tiled_gemm_kernel_v2<BM,BN,BK,TM,TN><<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(C.data(), dC, C.size()*sizeof(float), cudaMemcpyDeviceToHost));
}

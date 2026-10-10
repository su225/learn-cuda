#include <vector>

#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>

#include "common.h"

namespace cg = cooperative_groups;

template <int B, int M, int S>
__device__ __forceinline__
int swizzle(int x) {
    int source_mask = (1<<B)-1;
    static_assert(S > 0, "only positive distance is supported");
    return x ^ (((x >> (M+S)) & source_mask) << M);
}

template <int BM, int BN, int BK, int TM, int TN>
__global__
void tiled_gemm_async_swizzle_b_kernel(float const* __restrict__ A,
                                       float const* __restrict__ B,
                                       float *C,
                                       const int M,
                                       const int N,
                                       const int K) {
    constexpr int VEC = 4;
    constexpr int STORE_VEC = 4;

    static_assert(BM > 0 && BN > 0 && BK > 0, "Tile dimensions must be positive");
    static_assert(TM > 0 && TN > 0, "Thread tile dimensions must be positive");
    static_assert(BM % TM == 0, "BM must be divisible by TM");
    static_assert(BN % TN == 0, "BN must be divisible by TN");

    constexpr int THREADS_M = BM / TM;
    constexpr int THREADS_N = BN / TN;
    static_assert(THREADS_M * THREADS_N <= 1024, "Too many threads per block");
    static_assert(BK % VEC == 0, "BK must be divisible by float4 width");
    static_assert(BN % VEC == 0, "BN must be divisible by float4 width");
    static_assert(TN % STORE_VEC == 0, "TN must be divisible by float4 width");

    // When vectorized, instead of loading 1 column, we load VEC columns
    // with a single instruction from Global.
    constexpr int A_VEC_COLS = BK / VEC;
    constexpr int B_VEC_COLS = BN / VEC;

    auto block = cg::this_thread_block();

    const auto thread_index = block.thread_index();
    const auto block_index = block.group_index();

    const int tx = static_cast<int>(thread_index.x);
    const int ty = static_cast<int>(thread_index.y);
    const int tid = static_cast<int>(block.thread_rank());

    // Shared memory tiles. 128B alignment is useful for future vectorized/TMA
    // (Tensor Memory Accelerator) implementations.
    __shared__ alignas(128) float As[BM][BK];
    __shared__ alignas(128) float Bs[BK][BN];

    // The top-left (row,col) coordinates of the resulting submatrix
    const int block_row = static_cast<int>(block_index.y) * BM;
    // the current block/CTA is responsible for computing
    const int block_col = static_cast<int>(block_index.x) * BN;

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
    auto pipeline = cuda::make_pipeline();

    // Iterate over K in BK-sized chunks.
    for (int k0 = 0; k0 < K; k0 += BK) {
        pipeline.producer_acquire();

        // Load tile A. All threads in the block load consecutive addresses. This
        // coalesces memory accesses assuming row-major storage. Each thread with
        // tid=T would load (T, T+B, T+2B,...) where B is the number of threads.
        for (int vec_index = tid; vec_index < BM * A_VEC_COLS; vec_index += num_block_threads) {
            const int smem_row = vec_index / A_VEC_COLS;
            const int smem_vec_col = vec_index % A_VEC_COLS;
            const int smem_col = smem_vec_col * VEC;
            const int global_row = block_row + smem_row;
            const int global_col = k0 + smem_col;
            auto* dst = reinterpret_cast<float4*>(&As[smem_row][smem_col]);
            if (global_row < M && global_col < K) {
                const auto *src = reinterpret_cast<const float4*>(A + global_row*K + global_col);
                cuda::memcpy_async(dst, src, cuda::aligned_size_t<16>(sizeof(float4)), pipeline);
            } else {
                *dst = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }

        for (int vec_index = tid; vec_index < BK * B_VEC_COLS; vec_index += num_block_threads) {
            const int smem_row = vec_index / B_VEC_COLS;
            const int smem_vec_col = vec_index % B_VEC_COLS;
            const int smem_col = smem_vec_col * VEC;
            const int smem_physical_col = swizzle<1,2,3>(smem_col);
            const int global_row = k0 + smem_row;
            const int global_col = block_col + smem_col;
            auto* dst = reinterpret_cast<float4*>(&Bs[smem_row][smem_physical_col]);
            if (global_row < K && global_col < N) {
                const auto *src = reinterpret_cast<const float4*>(B + global_row*N + global_col);
                cuda::memcpy_async(dst, src, cuda::aligned_size_t<16>(sizeof(float4)), pipeline);
            } else {
                *dst = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        pipeline.producer_commit();
        pipeline.consumer_wait();
        pipeline.consumer_release();

        block.sync();

        const int valid_k = (BK < (K-k0)) ? BK : (K-k0);
        for (int k = 0; k < valid_k; ++k) {
            float b_frag[TN];
            constexpr int VECTORIZED_N = (TN / VEC) * VEC;

            #pragma unroll
            for (int n = 0; n < VECTORIZED_N; n += VEC) {
                const int logical_col = tx * TN + n;
                const int physical_col = swizzle<1, 2, 3>(logical_col);
                const float4 b = *reinterpret_cast<const float4*>(&Bs[k][physical_col]);

                b_frag[n + 0] = b.x;
                b_frag[n + 1] = b.y;
                b_frag[n + 2] = b.z;
                b_frag[n + 3] = b.w;
            }

            #pragma unroll
            for (int n = VECTORIZED_N; n < TN; ++n) {
                const int logical_col = tx * TN + n;
                const int physical_col = swizzle<1, 2, 3>(logical_col);
                b_frag[n] = Bs[k][physical_col];
            }

            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                const float a = As[ty * TM + m][k];
                #pragma unroll
                for (int n = 0; n < TN; ++n) {
                    acc[m][n] += a * b_frag[n];
                }
            }
        }
        block.sync();
    }

    // Store the register tile
    const int max_valid_m = TM < (M-threadblock_row) ? TM : (M-threadblock_row);
    const int max_valid_n = TN < (N-threadblock_col) ? TN : (N-threadblock_col);

    if (threadblock_col + TN <= N) {
        #pragma unroll
        for (int m = 0; m < max_valid_m; m++) {
            const int global_row = threadblock_row + m;
            #pragma unroll
            for (int n = 0; n < TN; n += STORE_VEC) {
                const float4 value = make_float4(acc[m][n], acc[m][n+1], acc[m][n+2], acc[m][n+3]);
                auto* dst = reinterpret_cast<float4*>(C + global_row*N + threadblock_col + n);
                *dst = value;
            }
        }
    } else {
        #pragma unroll
        for (int m = 0; m < max_valid_m; m++) {
            const int global_row = threadblock_row + m;
            #pragma unroll
            for (int n = 0; n < max_valid_n; n++) {
                const int global_col = threadblock_col + n;
                C[global_row * N + global_col] = acc[m][n];
            }
        }
    }
}

void tiled_gemm_async_swizzle_b(
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

    tiled_gemm_async_swizzle_b_kernel<BM,BN,BK,TM,TN><<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(C.data(), dC, C.size()*sizeof(float), cudaMemcpyDeviceToHost));
}

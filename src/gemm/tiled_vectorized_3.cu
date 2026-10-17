#include <vector>

#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/swizzle.hpp>

#include "common.h"

namespace cg = cooperative_groups;

template <int B, int M, int S>
__device__ __forceinline__
int swizzle(int x) {
    constexpr int source_mask = (1<<B)-1;
    static_assert(S > 0, "only positive distance is supported");
    return x ^ (((x >> (M+S)) & source_mask) << M);
}

template <int BM, int BN, int BK>
struct SmemLayoutPolicy {
    static_assert(BM == 128 && BN == 128, "Supports only 128x128 block");

    using ALayout = decltype(
        cute::composition(
            cute::Swizzle<3,2,7>{},
            cute::Layout<
                cute::Shape<cute::Int<BM>, cute::Int<BK>>,
                cute::Stride<cute::Int<1>, cute::Int<BM>>
            >{}
        )
    );

    using BLayout = decltype(
        cute::composition(
            cute::Swizzle<1,2,3>{},
            cute::Layout<
                cute::Shape<cute::Int<BK>, cute::Int<BN>>,
                cute::Stride<cute::Int<BN>, cute::Int<1>>
            >{}
        )
    );
};

template <int TN>
struct BFragmentLoader {
    template <class BTensor>
    __device__ __forceinline__
    static void load(
        const BTensor& sB,
        const int k,
        const int col,
        float (&b_frag)[TN]) {

        #pragma unroll
        for (int n = 0; n < TN; ++n) {
            b_frag[n] = sB(k, col+n);
        }
    }
};

template <>
struct BFragmentLoader<8> {
    template <class BTensor>
    __device__ __forceinline__
    static void load(
        const BTensor& sB,
        const int k,
        const int col,
        float (&b_frag)[8])
    {
        const float4 b0 = *reinterpret_cast<const float4*>(&sB(k, col+0));
        const float4 b1 = *reinterpret_cast<const float4*>(&sB(k, col+4));

        b_frag[0] = b0.x;
        b_frag[1] = b0.y;
        b_frag[2] = b0.z;
        b_frag[3] = b0.w;

        b_frag[4] = b1.x;
        b_frag[5] = b1.y;
        b_frag[6] = b1.z;
        b_frag[7] = b1.w;
    }
};

template <int TM>
struct AFragmentLoader {
    template <class ATensor>
    __device__ __forceinline__
    static void load(
        const ATensor& sA,
        const int k,
        const int row,
        float (&frag)[TM])
    {
        #pragma unroll
        for (int m = 0; m < TM; m++) {
            frag[m] = sA(row+m, k);
        }
    }
};

template <>
struct AFragmentLoader<8> {
    template <class ATensor>
    __device__ __forceinline__
    static void load(
        const ATensor& sA,
        int k,
        int row,
        float (&frag)[8])
    {
        const float4 a0 = *reinterpret_cast<const float4*>(&sA(row+0, k));
        const float4 a1 = *reinterpret_cast<const float4*>(&sA(row+4, k));

        frag[0] = a0.x;
        frag[1] = a0.y;
        frag[2] = a0.z;
        frag[3] = a0.w;

        frag[4] = a1.x;
        frag[5] = a1.y;
        frag[6] = a1.z;
        frag[7] = a1.w;
    }
};

template <int TM, int TN>
struct Multiplier {
    __device__ __forceinline__
    static void multiply(
        const float (&a)[TM],
        const float (&b)[TN],
        float (&acc)[TM][TN]
    ) {
        #pragma unroll
        for (int m = 0; m < TM; ++m) {
            const float av = a[m];
            #pragma unroll
            for (int n = 0; n < TN; ++n) {
                acc[m][n] += av * b[n];
            }
        }
    }
};

template <int BM, int BK, class GlobalATensor, class SharedATensor, int VEC = 4>
__device__ __forceinline__ void stage_a(
    GlobalATensor const& gA,
    SharedATensor sA,
    int tid, int num_threads
) {
    static_assert(BK % VEC == 0);
    constexpr int VECTORS_PER_ROW = BK / VEC;
    constexpr int NUM_VECTORS     = BM * VECTORS_PER_ROW;

    for (int i = tid; i < NUM_VECTORS; i += num_threads) {
        const int m  = i / VECTORS_PER_ROW;
        const int k4 = (i % VECTORS_PER_ROW) * VEC;
        const float4 value = *reinterpret_cast<const float4*>(&gA(m, k4));
        sA(m, k4 + 0) = value.x;
        sA(m, k4 + 1) = value.y;
        sA(m, k4 + 2) = value.z;
        sA(m, k4 + 3) = value.w;
    }
}

template <int BN, int BK, class GlobalBTensor, class SharedBTensor, class Pipeline, int VEC = 4>
__device__ __forceinline__ void stage_b_async(
    GlobalBTensor const& gB,
    SharedBTensor sB,
    int tid, int num_threads,
    Pipeline& pipeline
) {
    static_assert(BN % VEC == 0);
    constexpr int VECTORS_PER_ROW = BN / VEC;
    constexpr int NUM_VECTORS     = BK * VECTORS_PER_ROW;
    for (int i = tid; i < NUM_VECTORS; i += num_threads) {
        const int k  = i / VECTORS_PER_ROW;
        const int n4 = (i % VECTORS_PER_ROW) * VEC;
        cuda::memcpy_async(
            &sB(k, n4),
            &gB(k, n4),
            cuda::aligned_size_t<16>(sizeof(float4)),
            pipeline
        );
    }
}

template <int BM, int BN, int BK, int TM, int TN>
__global__
void tiled_gemm_vectorized_kernel_v3(float const* __restrict__ A,
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

    // Global tensors
    auto mA = cute::make_tensor(
        cute::make_gmem_ptr(A),
        cute::make_layout(
            cute::make_shape(M, K),
            cute::make_stride(K, cute::Int<1>{})
        )
    );

    auto mB = cute::make_tensor(
        cute::make_gmem_ptr(B),
        cute::make_layout(
            cute::make_shape(K, N),
            cute::make_stride(N, cute::Int<1>{})
        )
    );

    auto mC = cute::make_tensor(
        cute::make_gmem_ptr(C),
        cute::make_layout(
            cute::make_shape(M, N),
            cute::make_stride(N, cute::Int<1>{})
        )
    );

    using LayoutPolicy = SmemLayoutPolicy<BM,BN,BK>;
    using ASmemLayout = typename LayoutPolicy::ALayout;
    using BSmemLayout = typename LayoutPolicy::BLayout;

    __shared__ alignas(128) float sA_storage[cute::cosize_v<ASmemLayout>];
    __shared__ alignas(128) float sB_storage[cute::cosize_v<BSmemLayout>];

    auto sA = cute::make_tensor(cute::make_smem_ptr(sA_storage), ASmemLayout{});
    auto sB = cute::make_tensor(cute::make_smem_ptr(sB_storage), BSmemLayout{});

    auto block = cg::this_thread_block();
    const auto thread_index = block.thread_index();
    const auto block_index = block.group_index();
    auto gC = cute::local_tile(
        mC,
        cute::Shape<cute::Int<BM>, cute::Int<BN>>{},
        cute::make_coord(static_cast<int>(block_index.y),
                         static_cast<int>(block_index.x))
    );

    const int tx = static_cast<int>(thread_index.x);
    const int ty = static_cast<int>(thread_index.y);
    const int tid = static_cast<int>(block.thread_rank());

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
    const int num_k_tiles = (K + BK - 1) / BK;
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        auto gA = cute::local_tile(mA,
            cute::Shape<cute::Int<BM>, cute::Int<BK>>{},
            cute::make_coord(static_cast<int>(block_index.y), k_tile));

        auto gB = cute::local_tile(mB,
            cute::Shape<cute::Int<BK>, cute::Int<BN>>{},
            cute::make_coord(k_tile, static_cast<int>(block_index.x)));

        pipeline.producer_acquire();
        stage_b_async<BN,BK>(gB, sB, tid, num_block_threads, pipeline);
        pipeline.producer_commit();

        stage_a<BM,BK>(gA, sA, tid, num_block_threads);

        pipeline.consumer_wait();
        pipeline.consumer_release();
        block.sync(); // -- wait until staging is finished

        for (int k = 0; k < BK; ++k) {
            float b_frag[TN], a_frag[TM];
            AFragmentLoader<TM>::load(sA, k, ty*TM, a_frag);
            BFragmentLoader<TN>::load(sB, k, tx*TN, b_frag);
            Multiplier<TM,TN>::multiply(a_frag, b_frag, acc);
        }
        block.sync(); // -- wait until current multiplication is finished
    }

    // Store the register tile
    const int local_m = ty * TM;
    const int local_n = tx * TN;

    const int global_m_base = static_cast<int>(block_index.y) * BM;
    const int global_n_base = static_cast<int>(block_index.x) * BN;

    #pragma unroll
    for (int m = 0; m < TM; m++) {
        const int global_m = global_m_base + local_m + m;
        if (global_m >= M) continue;

        #pragma unroll
        for (int n = 0; n < TN; n++) {
            const int global_n = global_n_base + local_n + n;
            if (global_n < N) {
                gC(local_m + m, local_n + n) = acc[m][n];
            }
        }
    }
}

void tiled_gemm_vectorized_v3(
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

    tiled_gemm_vectorized_kernel_v3<BM,BN,BK,TM,TN><<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(C.data(), dC, C.size()*sizeof(float), cudaMemcpyDeviceToHost));
}

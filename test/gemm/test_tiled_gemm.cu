#include <gtest/gtest.h>
#include <vector>

#include "kernels/gemm.h"

#include "common.h"
#include "test_utils.h"
#include "gemm_test_case.h"

TEST_P(GemmTest, TiledKernel)
{
    auto tc = GetParam();

    int M = tc.M;
    int N = tc.N;
    int K = tc.K;

    std::vector<float> A = random_matrix(M * K, 1.0, 42);
    std::vector<float> B = random_matrix(K * N, 1.0, 42);
    std::vector<float> C_actual(M * N, 0);
    std::vector<float> C_expected(M * N, 0);

    cuBLASLt_gemm(A, B, C_expected, M, N, K);
    tiled_gemm(A, B, C_actual, M, N, K);

    compare(C_actual, C_expected, M, N, 1e-4f);
}

TEST_P(GemmTest, TiledKernelV2)
{
    auto tc = GetParam();

    int M = tc.M;
    int N = tc.N;
    int K = tc.K;

    std::vector<float> A = random_matrix(M * K, 1.0, 42);
    std::vector<float> B = random_matrix(K * N, 1.0, 42);
    std::vector<float> C_actual(M * N, 0);
    std::vector<float> C_expected(M * N, 0);

    cuBLASLt_gemm(A, B, C_expected, M, N, K);
    tiled_gemm_v2(A, B, C_actual, M, N, K);
    // for (int r = 0; r < M; ++r) {
    //     for (int c = 0; c < N; ++c) {
    //         float x = 0.0;
    //         for (int k = 0; k < K; ++k)
    //             x += A[r*K+k] * B[k*N+c];
    //         if (std::fabs(C_actual[r*N+c]-x) > 1e-4f) {
    //             printf("BAD r=%d c=%d actual=%f cpu=%f\n",
    //                 r, c, C_actual[r*N+c], x);
    //         }
    //     }
    // }

    compare(C_actual, C_expected, M, N, 1e-4f);
}

TEST_P(GemmTest, TiledKernelVectorizedV2)
{
    auto tc = GetParam();

    int M = tc.M;
    int N = tc.N;
    int K = tc.K;

    std::vector<float> A = random_matrix(M * K, 1.0, 42);
    std::vector<float> B = random_matrix(K * N, 1.0, 42);
    std::vector<float> C_actual(M * N, 0);
    std::vector<float> C_expected(M * N, 0);

    cuBLASLt_gemm(A, B, C_expected, M, N, K);
    tiled_gemm_vectorized_v2(A, B, C_actual, M, N, K);

    compare(C_actual, C_expected, M, N, 1e-4f);
}

INSTANTIATE_TEST_SUITE_P(
    Correctness,
    GemmTest,
    ::testing::Values(
        GemmCase{.M = 128, .N = 128, .K = 128},
        GemmCase{.M = 512, .N = 512, .K = 512},
        GemmCase{.M = 1024, .N = 1024, .K = 1024},
        GemmCase{.M = 2048, .N = 2048, .K = 2048}
    )
);

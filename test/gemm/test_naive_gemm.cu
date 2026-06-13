#include <gtest/gtest.h>
#include <vector>

#include "kernels/gemm.h"

#include "test_utils.h"
#include "random_matrix.h"
#include "gemm_test_case.h"

TEST_P(GemmTest, NaiveKernel)
{
    auto tc = GetParam();

    int M = tc.M;
    int N = tc.N;
    int K = tc.K;

    std::vector<float> A = random_matrix(M * K);
    std::vector<float> B = random_matrix(K * N);
    std::vector<float> C_actual(M * N, 0);
    std::vector<float> C_expected(M * N, 0);

    naive_gemm(A, B, C_actual, M, N, K);
    cuBLASLt_gemm(A, B, C_expected, M, N, K);

    compare(C_actual, C_expected, 1e-4f);
}

INSTANTIATE_TEST_SUITE_P(
    Correctness,
    GemmTest,
    ::testing::Values(
        GemmCase{.M = 128, .N = 128, .K = 128},
        GemmCase{.M = 512, .N = 512, .K = 512},
        GemmCase{.M = 1024, .N = 1024, .K = 1024}
    )
);

#include <gtest/gtest.h>
#include <vector>

#include "test_utils.h"
#include "kernels/gemm.h"

void run_gemm_test(int M, int N, int K)
{
    std::vector<float> A = random_matrix(M * K);
    std::vector<float> B = random_matrix(K * N);
    std::vector<float> C_actual(M * N, 0);
    std::vector<float> C_expected(M * N, 0);

    naive_gemm(A, B, C_actual, M, N, K);
    cuBLASLt_gemm(A, B, C_expected, M, N, K);

    compare(C_actual, C_expected, 1e-4f);
}

TEST(GEMM, Naive_128x128)
{
    run_gemm_test(128, 128, 128);
}

TEST(GEMM, Naive_1024x1024) {
    run_gemm_test(1024, 1024, 1024);
}

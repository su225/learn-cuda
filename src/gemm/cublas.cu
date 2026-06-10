#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cublasLt.h>

#include "common.h"

void cuBLASLt_gemm(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int M,
    int N,
    int K
) {
   	float *d_A, *d_B, *d_C;
	CUDA_CHECK(cudaMalloc(&d_A, A.size() * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_B, B.size() * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_C, C.size() * sizeof(float)));

	CUDA_CHECK(cudaMemcpy(d_A, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_B, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice));

	cublasLtHandle_t handle;
	cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
	cublasLtMatmulDesc_t opDesc;

    CUBLAS_CHECK(cublasLtCreate(&handle));
	CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, M, K, K));
	CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, K, N, N));
	CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, M, N, N));

	CUBLAS_CHECK(cublasLtMatmulDescCreate(
		&opDesc,
		CUBLAS_COMPUTE_32F,
		CUDA_R_32F));

	cublasOperation_t opN = CUBLAS_OP_N;
	CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)));
	CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

	float alpha = 1.0f;
	float beta = 0.0f;

	cublasLtMatmulPreference_t pref;
	CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));

	size_t workspaceSize = 1 << 20;
	void *workspace;
	CUDA_CHECK(cudaMalloc(&workspace, workspaceSize));

	CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
		pref,
		CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
		&workspaceSize,
		sizeof(workspaceSize)));

	cublasLtMatmulHeuristicResult_t heuristic;
	int returned = 0;

	CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
		handle,
		opDesc,
		Adesc,
		Bdesc,
		Cdesc,
		Cdesc,
		pref,
		1,
		&heuristic,
		&returned));
	if (returned == 0) {
	    throw std::runtime_error("No algorithm found");
	}

	// -- actual GEMM --
    CUBLAS_CHECK(cublasLtMatmul(
        handle,
        opDesc,
        &alpha,
        d_A,
        Adesc,
        d_B,
        Bdesc,
        &beta,
        d_C,
        Cdesc,
        d_C,
        Cdesc,
        &heuristic.algo,
        workspace,
        workspaceSize,
        0));

	cudaMemcpy(C.data(), d_C, C.size() * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < 15; i++) {
    	for (int j = 0; j < 15; j++) {
        	std::cout << std::fixed << std::setprecision(4)
            	      << C[INDEX(i,j,M,N)] << " ";
    	}
    	std::cout << std::endl;
	}

	cudaFree(workspace);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cublasLtDestroy(handle);
}

#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cublasLt.h>

#define CUDA_CHECK(call)                                                   \
do {                                                                       \
	cudaError_t error = (call);                                              \
	if (error != cudaSuccess) {                                            \
		std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
				  << " code=" << static_cast<int>(error)                   \
				  << " reason=" << cudaGetErrorString(error) << std::endl; \
		std::exit(EXIT_FAILURE);                                           \
	}                                                                      \
} while (0);

#define CUBLAS_CHECK(call) \
do { \
	cublasStatus_t st = (call); \
	if (st != CUBLAS_STATUS_SUCCESS) { \
		std::cerr << "cuBLASLt error: " << st << std::endl; \
		std::exit(EXIT_FAILURE); \
	} \
} while (0);

#define INDEX(i,j,r,c) ((i*c + j))

int main() {
	int M = 1024;
	int N = 1024;
	int K = 1024;

	std::vector<float> A(M * K);
	std::vector<float> B(K * N);
	std::vector<float> C(M * N);
	for (int i = 0; i < M; i++)
		for (int j = 0; j < K; j++)
			A[INDEX(i,j,M,K)] = 1.0*(i+j)/(M+K);

	for (int i = 0; i < K; i++)
		for (int j = 0; j < N; j++)
			B[INDEX(i,j,K,N)] = 1.0*(i+j)/(K+N);

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
		std::cerr << "No algorithm found" << std::endl;
		return EXIT_FAILURE;
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
    return 0;
}

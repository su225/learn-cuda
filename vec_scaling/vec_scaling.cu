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

__global__ void vec_scaling(float *A, float scale, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        A[i] = A[i] * scale;
    }
}

int main() {
    int N = 1024 * 1024;
    float scale = 1.5;
    size_t size = N * sizeof(float);
    
    float *h_A;
    float *d_A;
    
    h_A = (float *)malloc(size);
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
    }

    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));

    // Launch the kernel
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    vec_scaling<<<blocks, threads>>>(d_A, scale, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost));
    
    // Verification
    for (int i = 0; i < 5; i++) {
        printf("%f ", h_A[i]);
    }
    printf("\n");
    
    CUDA_CHECK(cudaFree(d_A));
    free(h_A);
    
    return 0;
}

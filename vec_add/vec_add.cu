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

__global__ void vec_add(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = 1024 * 1024 * 1024;
    size_t size = N * sizeof(float);
    
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    
    // Allocate host memory
    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);
    
    // Initialize host memory
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = 2 * i;
    }
    
    // Allocate GPU memory in the global memory area.
    // The actual SM-level memory allocation is done
    // later inside the GPU, but not here. This is because
    // the blocks are allocated to a Streaming Multiprocessor
    // only after the kernel is launched.
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    
    // Launch the kernel
    const int threads = 1024;
    const int blocks = (N + threads - 1) / threads;
    vec_add<<<blocks, threads>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    
    // Copy the results back from the GPU
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    
    // Verification
    for (int i = 0; i < 5; i++) {
        printf("%f ", h_C[i]);
    }
    printf("\n");
    
    CUDA_CHECK(cudaFree(d_A)); 
    CUDA_CHECK(cudaFree(d_B)); 
    CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C);
    
    return 0;
}

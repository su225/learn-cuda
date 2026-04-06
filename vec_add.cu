#include <stdio.h>

__global__ void vec_add(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = 1024;
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
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // Launch the kernel
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    vec_add<<<blocks, threads>>>(d_A, d_B, d_C, N);
    
    // Copy the results back from the GPU
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // Verification
    for (int i = 0; i < 5; i++) {
        printf("%f ", h_C[i]);
    }
    printf("\n");
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    
    return 0;
}

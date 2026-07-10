#include <vector>
#include <cuda_runtime.h>

#include "common.h"

template <unsigned int x> inline constexpr int log2_v = 1 + log2_v<(x >> 1)>;
template <> inline constexpr int log2_v<0> = 1;

const unsigned int THREADS_PER_WARP = 32;
const unsigned int LOG2_THREADS_PER_WARP = 5;

// Define block, warp, thread level dimensions
const unsigned int BM = 128;
const unsigned int BN = 128;
const unsigned int BK = 128;
const unsigned int rowsPerWarp = 4;
const unsigned int colsPerWarp = 128;
const unsigned int rowsPerLane = 4;
const unsigned int colsPerLane = 4;

const unsigned int warpRows = BM / rowsPerWarp;
const unsigned int warpCols = BN / colsPerWarp;
const unsigned int laneRows = rowsPerWarp / rowsPerLane;
const unsigned int laneCols = colsPerWarp / colsPerLane;

// Define dimensions for loading the (warp, lane, value) for A
const unsigned int LOAD_WARP_A_DIM_ROWS = 4;
const unsigned int LOAD_WARP_A_DIM_COLS = 128;
const unsigned int LOAD_WARP_A_ROWS = BM / LOAD_WARP_A_DIM_ROWS;
const unsigned int LOAD_WARP_A_COLS = BK / LOAD_WARP_A_DIM_COLS;
const unsigned int LOAD_LANE_A_DIM_ROWS = 1;
const unsigned int LOAD_LANE_A_DIM_COLS = 16;
const unsigned int LOAD_LANE_A_ROWS = LOAD_WARP_A_DIM_ROWS / LOAD_LANE_A_DIM_ROWS;
const unsigned int LOAD_LANE_A_COLS = LOAD_WARP_A_DIM_COLS / LOAD_LANE_A_DIM_COLS;

// Define dimensions for loading the (warp, lane, value) for B
const unsigned int LOAD_WARP_B_DIM_ROWS = 4;
const unsigned int LOAD_WARP_B_DIM_COLS = 128;
const unsigned int LOAD_WARP_B_ROWS = BM / LOAD_WARP_B_DIM_ROWS;
const unsigned int LOAD_WARP_B_COLS = BK / LOAD_WARP_B_DIM_COLS;
const unsigned int LOAD_LANE_B_DIM_ROWS = 1;
const unsigned int LOAD_LANE_B_DIM_COLS = 16;
const unsigned int LOAD_LANE_B_ROWS = LOAD_WARP_B_DIM_ROWS / LOAD_LANE_B_DIM_ROWS;
const unsigned int LOAD_LANE_B_COLS = LOAD_WARP_B_DIM_COLS / LOAD_LANE_B_DIM_COLS;


__global__ void tiled_gemm_kernel_v2(float const *A,
                                     float const *B,
                                     float *C,
                                     const int M,
                                     const int N,
                                     const int K) {
    __shared__ float sA[BM * BK];
    __shared__ float sB[BK * BN];
    float sC[BM * BN];

    const int blockIndex = blockIdx.x;
    const int blockRow = blockIndex >> log2_v<BN>;
    const int blockCol = blockIndex & (BN-1);

    const int warpIndex = threadIdx.x >> LOG2_THREADS_PER_WARP;
    const int laneIndex = threadIdx.x & (THREADS_PER_WARP-1);
    const int warpRow = warpIndex >> log2_v<colsPerWarp>;
    const int warpCol = warpIndex & (colsPerWarp - 1);
    const int laneRow = laneIndex >> log2_v<colsPerLane>;
    const int laneCol = laneIndex & (colsPerLane - 1);

    const int startRow = warpRow*rowsPerWarp + laneRow*rowsPerLane;
    const int startCol = warpCol*colsPerWarp + laneCol*colsPerLane;

    int numBatches = K / BK;
    for (int batchId = 0; batchId < numBatches; batchId++) {
        const int loadWarpARow = warpIndex >> log2_v<LOAD_WARP_A_COLS>;
        const int loadWarpACol = warpIndex & (LOAD_WARP_A_COLS - 1);
        const int loadLaneARow = laneIndex >> log2_v<LOAD_LANE_A_COLS>;
        const int loadLaneACol = laneIndex & (LOAD_LANE_A_COLS - 1);
        for (int i = 0; i < LOAD_LANE_A_DIM_ROWS; i++) {
            for (int j = 0; j < LOAD_LANE_A_DIM_COLS; j++) {
                int loadRow = (blockRow * BM) + (loadWarpARow * LOAD_WARP_A_DIM_ROWS) + (loadLaneARow) + i;
                int loadCol = (batchId*BK) + (blockCol * BK) + (loadLaneACol * LOAD_LANE_A_DIM_ROWS) + j;
                int storeRow = (loadWarpARow * LOAD_WARP_A_DIM_ROWS) + (loadLaneARow) + i;
                int storeCol = (loadLaneACol * LOAD_WARP_A_DIM_COLS) + j;
                if (loadRow < M && loadCol < K) {
                    float elem = A[loadRow*K + loadCol];
                    sA[storeRow * BK + storeCol] = elem;
                } else {
                    sA[storeRow * BK + storeCol] = 0.0;
                }
            }
        }

        const int loadWarpBRow = warpIndex >> log2_v<LOAD_WARP_B_COLS>;
        const int loadWarpBCol = warpIndex & (LOAD_WARP_B_COLS - 1);
        const int loadLaneBRow = laneIndex >> log2_v<LOAD_LANE_B_COLS>;
        const int loadLaneBCol = laneIndex & (LOAD_LANE_B_COLS - 1);
        for (int i = 0; i < LOAD_LANE_B_DIM_ROWS; i++) {
            for (int j = 0; j < LOAD_LANE_B_DIM_COLS; j++) {
                int loadRow = (batchId*BK) + (blockRow * BK) + (loadWarpBRow + LOAD_WARP_B_DIM_ROWS) + (loadLaneBRow) + i;
                int loadCol = (blockCol*BN) + (loadLaneBCol * LOAD_LANE_B_DIM_ROWS) + j;
                int storeRow = (loadWarpBRow * LOAD_WARP_B_DIM_ROWS) + (loadLaneBRow) + i;
                int storeCol = (loadLaneBCol * LOAD_WARP_B_DIM_COLS) + j;
                if (loadRow < K && loadCol < N) {
                    float elem = B[loadRow*K + loadCol];
                    sB[storeRow * BK + storeCol] = elem;
                } else {
                    sB[storeRow * BK + storeCol] = 0.0;
                }
            }
        }
        __syncthreads();

        for (int i = 0; i < rowsPerLane; i++) {
            for (int j = 0; j < colsPerLane; j++) {
                const int row = startRow + i;
                const int col = startCol + j;
                const int loc = row*BN + col;
                for (int k = 0; k < BK; k++) {
                    sC[loc] += sA[row*BK + k] * sB[k*BN + col];
                }
            }
        }
        __syncthreads();
    }
    int globalStartRow = blockRow*BM + warpRow*rowsPerWarp + laneRow*rowsPerLane;
    int globalStartCol = blockCol*BN + warpCol*colsPerWarp + laneCol*colsPerLane;
    for (int i = 0; i < rowsPerLane; i++) {
        for (int j = 0; j < colsPerLane; j++) {
            C[(globalStartRow+i)*N + (globalStartCol+j)] = sC[(startRow+i)*BN + (startCol+j)];
        }
    }
}

void tiled_gemm_v2(
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

    int totalElements = M * N;
    int threadsPerBlock = 32 * 32;
    int totalGridBlocks = (totalElements + threadsPerBlock - 1) / threadsPerBlock;
    tiled_gemm_kernel_v2<<<totalGridBlocks, threadsPerBlock>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(C.data(), dC, C.size()*sizeof(float), cudaMemcpyDeviceToHost));
}

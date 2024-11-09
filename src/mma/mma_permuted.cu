// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:02:28 on Tue, Feb 28, 2023
//
// Description: mma permuted hgemm

#include "common.h"
#include "tensor_core_util.cuh"

__global__ void mmaPermutedKernel(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C,
                                  size_t M, size_t N, size_t K) {

    // m16n8k16
    const size_t M_tiles = ceil(M, MMA_M);
    const size_t N_tiles = ceil(N, MMA_N);
    const size_t K_tiles = ceil(K, MMA_K);

    extern __shared__ half smem[][AB_SMEM_STRIDE];
    half* smemC = &smem[0][0];
    half* smemA = &smem[0][0];
    half* smemB = &smem[BLOCK_ROWS][0];

    const size_t warp_id = threadIdx.x>>5;  // warp_id=0,1,...,7
    const size_t lane_id = threadIdx.x&31;

    // RC[4][8][2];
    uint32_t RC[WARP_COL_TILES][WARP_ROW_TILES][2];
    memset(RC, 0, sizeof(RC));

    /* step 1: block swizzle (the same as base version) */
    size_t block_tile_i;
    size_t block_tile_j;
    swizzle(&block_tile_i, &block_tile_j);
    if (block_tile_i >= M_tiles || block_tile_j >= N_tiles) return;

    // get the address of exact row to put in SRAM
    const half *A_warp_ptr = &A(block_tile_i, warp_id); 
    const half *B_warp_ptr = &B(block_tile_j, warp_id);

    #pragma unroll
    for (size_t tile_k = 0; tile_k < K_tiles; tile_k += CHUNK_K) {

        /* step 2: load Matrix A & B from HBM to SRAM (deals with bank conflict) */
        ldgstsA_permute(warp_id, lane_id, A_warp_ptr, tile_k, K, smemA);
        ldgstsB_permute(warp_id, lane_id, B_warp_ptr, tile_k, K, smemB);

        __syncthreads();

        uint32_t RA[WARP_COL_TILES][4]; // RA[4][4]
        uint32_t RB[WARP_ROW_TILES][2]; // RA[8][2]

        #pragma unroll
        for (size_t k_step = 0; k_step < CHUNK_K; ++k_step) {

            /* step 3: load Matrix A & B from SRAM to Register */
            #pragma unroll
            for (size_t i = 0; i < WARP_COL_TILES; ++i)
                ldsA_permute(i, k_step, warp_id, lane_id, smemA, RA);

            #pragma unroll
            for (size_t j = 0; j < WARP_ROW_TILES; ++j)
                ldsB_permute(j, k_step, warp_id, lane_id, smemB, RB);

            /* step 4: calc mma C=A@B */
            #pragma unroll
            for (size_t i = 0; i < WARP_COL_TILES; ++i) {
                #pragma unroll
                for (size_t j = 0; j < WARP_ROW_TILES; ++j)
                    hm16n8k16(i, j, RA, RB, RC);
            }
        }

        __syncthreads();
    }

    /* step 5: load result from Register to SRAM */
    half *smem_warp_tile_row_ptr = smemC + ((warp_id>>1)<<13);

    //#pragma unroll
    for (size_t i = 0; i < WARP_COL_TILES; ++i) {
        #pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j)
            stsC_permute(i, j, warp_id, lane_id, smem_warp_tile_row_ptr, RC);
    }

    __syncthreads();

    const half *smem_warp_stream_ptr = smemC + warp_id * MMA_M * 2 * C_SMEM_STRIDE;
    const size_t gmem_idx = (block_tile_i + warp_id * 2) * MMA_M * N + block_tile_j * MMA_N;
    const half *src_gmem_warp_stream_ptr = &C[gmem_idx];

    /* step 6: load result from SRAM to HBM */
    #pragma unroll
    for (size_t i = 0; i < MMA_M; ++i)
        ldsC_permute(i, N, lane_id, src_gmem_warp_stream_ptr, smem_warp_stream_ptr);
}

size_t initMmaPermuted() {
    int dev_id = 0;
    HGEMM_CHECK_CUDART_ERROR(cudaGetDevice(&dev_id));

    cudaDeviceProp dev_prop;
    HGEMM_CHECK_CUDART_ERROR(cudaGetDeviceProperties(&dev_prop, dev_id));

    // #define BLOCK_ROWS 256
    // #define BLOCK_COLS 128
    // #define C_SMEM_STRIDE 128  // BLOCK_COLS
    size_t smem_max_size =
        std::max((BLOCK_ROWS + BLOCK_COLS) * AB_SMEM_STRIDE * sizeof(half), 
                  BLOCK_ROWS * C_SMEM_STRIDE * sizeof(half));
    HLOG("smem_max_size: %.0f KBytes (%zu Bytes)", static_cast<double>(smem_max_size) / 1024, smem_max_size);

    HGEMM_CHECK_GT(dev_prop.sharedMemPerMultiprocessor, smem_max_size);
    HGEMM_CHECK_CUDART_ERROR(
        cudaFuncSetAttribute(mmaPermutedKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max_size));

    return smem_max_size;
}

void mmaPermuted(half *A, half *B, half *C, size_t M, size_t N, size_t K) {
    static size_t smem_max_size = initMmaPermuted();

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(BLOCK_STRIDE, ceil(M, BLOCK_ROWS), ceil(N, BLOCK_COLS * BLOCK_STRIDE));

    mmaPermutedKernel<<<grid, block, smem_max_size>>>(A, B, C, M, N, K);
}

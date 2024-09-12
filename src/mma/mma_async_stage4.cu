// Modified Author: Raymond
// Description: mma async stage4 hgemm

#include "common.h"
#define K_STAGE 4

__global__ void mmaAsyncStage4Kernel(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C,
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

    uint32_t RA[2][WARP_COL_TILES][4]; // RA[2][4][4]
    uint32_t RB[2][WARP_ROW_TILES][2]; // RB[2][8][2]

    // RC[4][8][2];
    uint32_t RC[WARP_COL_TILES][WARP_ROW_TILES][2];
    memset(RC, 0, sizeof(RC));

    /* step 1: block swizzle (the same as base version) */
    size_t block_tile_i;
    size_t block_tile_j;
    swizzle(&block_tile_i, &block_tile_j);
    if (block_tile_i >= M_tiles || block_tile_j >= N_tiles) return;

    constexpr size_t B_smem_idx_off = BLOCK_ROWS;
    constexpr size_t smem_stage_off = BLOCK_ROWS + BLOCK_COLS;

    // get the address of exact row to put in SRAM
    const half *A_warp_ptr = &A(block_tile_i, warp_id); 
    const half *B_warp_ptr = &B(block_tile_j, warp_id);

    size_t smem_store_idx = 0;
    size_t smem_load_idx = 0;

    size_t smem_store_off = 0;
    size_t smem_load_off = 0;

    ldgstsA_stage(warp_id, lane_id, A_warp_ptr, 0, K, smemA, smem_store_off);
    ldgstsB_stage(warp_id, lane_id, B_warp_ptr, 0, K, smemB, smem_store_off);

    CP_ASYNC_COMMIT_GROUP();

    smem_store_idx = (smem_store_idx + 1) % K_STAGE;
    smem_store_off = smem_store_idx * smem_stage_off;

    ldgstsA_stage(warp_id, lane_id, A_warp_ptr, CHUNK_K, K, smemA, smem_store_off);
    ldgstsB_stage(warp_id, lane_id, B_warp_ptr, CHUNK_K, K, smemB, smem_store_off);

    CP_ASYNC_COMMIT_GROUP();

    smem_store_idx = (smem_store_idx + 1) % K_STAGE;
    smem_store_off = smem_store_idx * smem_stage_off;

    ldgstsA_stage(warp_id, lane_id, A_warp_ptr, 2*CHUNK_K, K, smemA, smem_store_off);
    ldgstsB_stage(warp_id, lane_id, B_warp_ptr, 2*CHUNK_K, K, smemB, smem_store_off);

    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP(2);

    __syncthreads();

    size_t reg_store_idx = 0;
    size_t reg_load_idx = 1;

    #pragma unroll
    for (size_t i = 0; i < WARP_COL_TILES; ++i)
        ldsA_stage2(i, 0, reg_store_idx, smem_load_off, warp_id, lane_id, smemA, RA);

    #pragma unroll
    for (size_t j = 0; j < WARP_ROW_TILES; ++j)
        ldsB_stage2(j, 0, reg_store_idx, smem_load_off, warp_id, lane_id, smemB, RB);

    #pragma unroll
    for (size_t tile_k = CHUNK_K * (K_STAGE - 1); tile_k < K_tiles; tile_k += CHUNK_K) {
        reg_store_idx ^= 1;
        reg_load_idx ^= 1;

        #pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i)
            ldsA_stage2(i, 1, reg_store_idx, smem_load_off, warp_id, lane_id, smemA, RA);

        #pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j)
            ldsB_stage2(j, 1, reg_store_idx, smem_load_off, warp_id, lane_id, smemB, RB);

        #pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
            #pragma unroll
            for (size_t j = 0; j < WARP_ROW_TILES; ++j)
                hm16n8k16(i, j, reg_load_idx, RA, RB, RC);
        }

        smem_store_idx = (smem_store_idx + 1) % K_STAGE;
        smem_store_off = smem_store_idx * smem_stage_off;

        ldgstsA_stage(warp_id, lane_id, A_warp_ptr, tile_k, K, smemA, smem_store_off);
        ldgstsB_stage(warp_id, lane_id, B_warp_ptr, tile_k, K, smemB, smem_store_off);

        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP(2);

        __syncthreads();

        reg_store_idx ^= 1;
        reg_load_idx ^= 1;

        smem_load_idx = (smem_load_idx + 1) % K_STAGE;
        smem_load_off = smem_load_idx * smem_stage_off;

        #pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i)
            ldsA_stage2(i, 0, reg_store_idx, smem_load_off, warp_id, lane_id, smemA, RA);

        #pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j)
            ldsB_stage2(j, 0, reg_store_idx, smem_load_off, warp_id, lane_id, smemB, RB);

        #pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
            #pragma unroll
            for (size_t j = 0; j < WARP_ROW_TILES; ++j)
                hm16n8k16(i, j, reg_load_idx, RA, RB, RC);
        }
    }

    #pragma unroll
    for (size_t k_step = 0; k_step < CHUNK_K; ++k_step) {
        reg_store_idx ^= 1;
        reg_load_idx ^= 1;

        #pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i)
            ldsA_stage3(i, k_step, reg_store_idx, smem_load_off, warp_id, lane_id, smemA, RA);

        #pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j)
            ldsB_stage3(j, k_step, reg_store_idx, smem_load_off, warp_id, lane_id, smemB, RB);

        #pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
            #pragma unroll
            for (size_t j = 0; j < WARP_ROW_TILES; ++j)
                hm16n8k16(i, j, reg_load_idx, RA, RB, RC);
        }

        if (k_step + 2 == CHUNK_K) {
            smem_load_idx = (smem_load_idx + 1) % K_STAGE;
            smem_load_off = smem_load_idx * smem_stage_off;

            CP_ASYNC_WAIT_GROUP(1);

            __syncthreads();
        }
    }

    #pragma unroll
    for (size_t k_step = 0; k_step < CHUNK_K; ++k_step) {
        reg_store_idx ^= 1;
        reg_load_idx ^= 1;

        #pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i)
            ldsA_stage3(i, k_step, reg_store_idx, smem_load_off, warp_id, lane_id, smemA, RA);

        #pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j)
            ldsB_stage3(j, k_step, reg_store_idx, smem_load_off, warp_id, lane_id, smemB, RB);

        #pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
            #pragma unroll
            for (size_t j = 0; j < WARP_ROW_TILES; ++j)
                hm16n8k16(i, j, reg_load_idx, RA, RB, RC);
        }

        if (k_step + 2 == CHUNK_K) {
            smem_load_idx = (smem_load_idx + 1) % K_STAGE;
            smem_load_off = smem_load_idx * smem_stage_off;

            CP_ASYNC_WAIT_GROUP(0);

            __syncthreads();
        }
    }

#pragma unroll
    for (size_t k_step = 1; k_step < CHUNK_K; ++k_step) {
        reg_store_idx ^= 1;
        reg_load_idx ^= 1;

        #pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i)
            ldsA_stage2(i, k_step, reg_store_idx, smem_load_off, warp_id, lane_id, smemA, RA);

        #pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j)
            ldsB_stage2(j, k_step, reg_store_idx, smem_load_off, warp_id, lane_id, smemB, RB);

        #pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
            #pragma unroll
            for (size_t j = 0; j < WARP_ROW_TILES; ++j)
                hm16n8k16(i, j, reg_load_idx, RA, RB, RC);
        }
    }

    #pragma unroll
    for (size_t i = 0; i < WARP_COL_TILES; ++i) {
        #pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j)
            hm16n8k16(i, j, reg_store_idx, RA, RB, RC);
    }

    __syncthreads();

    half *smem_warp_tile_row_ptr = smemC + (warp_id / BLOCK_ROW_WARPS) * C_SMEM_STRIDE * WARP_ROWS;
    #pragma unroll
    for (size_t i = 0; i < WARP_COL_TILES; ++i) {
        #pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j)
            stsC_permute(i, j, warp_id, lane_id, smem_warp_tile_row_ptr, RC);
    }

    __syncthreads();

    const half *smem_warp_stream_ptr = smemC + warp_id * MMA_M * 2 * C_SMEM_STRIDE;
    const size_t gmem_idx = (block_tile_i + warp_id * 2) * MMA_M * N + block_tile_j * MMA_N;
    const half *src_gmem_warp_stream_ptr = &C[gmem_idx];

    #pragma unroll
    for (size_t i = 0; i < MMA_M; ++i)
        ldsC_permute(i, N, lane_id, src_gmem_warp_stream_ptr, smem_warp_stream_ptr);
}

size_t initMmaAsyncStage4() {
    int dev_id = 0;
    HGEMM_CHECK_CUDART_ERROR(cudaGetDevice(&dev_id));

    cudaDeviceProp dev_prop;
    HGEMM_CHECK_CUDART_ERROR(cudaGetDeviceProperties(&dev_prop, dev_id));

    size_t smem_max_size = std::max((BLOCK_ROWS + BLOCK_COLS) * AB_SMEM_STRIDE * sizeof(half) * K_STAGE,
                                    BLOCK_ROWS * C_SMEM_STRIDE * sizeof(half));
    HLOG("smem_max_size: %.0f KBytes (%zu Bytes)", static_cast<double>(smem_max_size) / 1024, smem_max_size);

    HGEMM_CHECK_GT(dev_prop.sharedMemPerMultiprocessor, smem_max_size);
    HGEMM_CHECK_CUDART_ERROR(
        cudaFuncSetAttribute(mmaAsyncStage4Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max_size));

    return smem_max_size;
}

void mmaAsyncStage4(half *A, half *B, half *C, size_t M, size_t N, size_t K) {
    static size_t smem_max_size = initMmaAsyncStage4();

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(BLOCK_STRIDE, ceil(M, BLOCK_ROWS), ceil(N, BLOCK_COLS * BLOCK_STRIDE));

    mmaAsyncStage4Kernel<<<grid, block, smem_max_size>>>(A, B, C, M, N, K);
}

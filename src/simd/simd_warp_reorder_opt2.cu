// Copyright 2023. All Rights Reserved.
// Author: Raymond
// Date: 16:27:28 on Fri, Jun 07, 2024
//
// Description: simd warp reorder opt-2

#include "common.h"

#define THREADS_PER_BLOCK 256

#define BLOCK_ROWS 128
#define BLOCK_COLS 128
#define AB_SMEM_STRIDE 16

#define MMA_M 8
#define MMA_N 8
#define MMA_K 8
#define CHUNK_K 2

#define WARP_ROW_TILES 4
#define WARP_COL_TILES 8

#define tileA(i,j) tileA[(i)*(lda)+(j)]
#define tileB(i,j) tileB[(j)*(ldb)+(i)]
#define tileC(i,j) tileC[(i)*(ldc)+(j)]

#define sharedA(i) sharedA[(i)*MMA_K]
#define sharedB(i) sharedB[(i)*MMA_K]
#define sharedMem(i,j) sharedMem[(i)*(BLOCK_COLS)+(j)]

#define A(i,j) A[(i)*lda+(j)]
#define B(i,j) B[(i)+(j)*ldb]
#define C(i,j) C[(i)*ldc+(j)]

__device__ void ldgstsAOpt2(half* sharedA, const half* tileA, 
                            int tile_row_a, int tile_col_a, 
                            int smem_row_a, int lda) {
    /* 
       tile_row_a = 0,1,...,127, 0,1,...,127
       tile_col_a = 0,0,...,0,   1,1,...,1
     */
    *((float4*)(&sharedA(smem_row_a))) = 
            *((float4*)(&tileA(tile_row_a, tile_col_a)));
}

__device__ void ldgstsBOpt2(half* sharedB, const half* tileB, 
                            int tile_row_b, int tile_col_b, 
                            int smem_col_b, int ldb) {
    /* 
       tile_row_b = 0,0,...,0,   1,1,...,1
       tile_col_b = 0,1,...,127, 0,1,...,127
     */
    *((float4*)(&sharedB(smem_col_b))) = 
            *((float4*)(&tileB(tile_row_b, tile_col_b)));
}

__global__ void simdWarpReorderKernelOpt2(const half *__restrict__ A, 
                                      const half *__restrict__ B, 
                                      half *__restrict__ C, 
                                      size_t M, size_t N, size_t K) {

    int lda=M, ldb=K, ldc=M;
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;

    int warp_id = tx>>5;
    int lane_id = tx&31;

    /*
       every block has 8 warps: 4x2
       warp.shape = (4,2) ==> warp_row=0,1,2,3; warp_col=0,1
     */
    int warp_row = warp_id & 3, warp_col = warp_id >> 2;

    /* 
       tile_row_a = 0,1,...,127, 0,1,...,127
       tile_col_a = 0,0,...,0,   1,1,...,1
     */
    int tile_row_a = tx&127, tile_col_a = (tx>>7)<<3;

    /* 
       smem_row_a = 0,1,...,255
     */
    int smem_row_a = tx;

    /* 
       tile_row_b = 0,0,...,0,   1,1,...,1
       tile_col_b = 0,1,...,127, 0,1,...,127
     */
    int tile_row_b = (tx>>7)<<3, tile_col_b = tx&127;

    /* 
       smem_col_b = 0,1,...,255
     */
    int smem_col_b = tx;

    /*
       warp_row_c = 32*(0,1,2,3)
       warp_col_c = 64*(0,1)
     */
    int warp_row_c = warp_row<<5;
    int warp_col_c = warp_col<<6;

    /*
       warp_tid_row = 8*(0,0,0,0,0,0,0,0| ... | 3,3,3,3,3,3,3,3)
       warp_tid_col = (0,1,2,3,4,5,6,7| ... | 0,1,2,3,4,5,6,7)
     */
    int warp_tid_row = (lane_id>>3)<<3;
    int warp_tid_col = lane_id&7;

    // shared memory for input matrix A & B
    extern __shared__ half sharedMem[];
    half* sharedA = sharedMem; // 256x8
    half* sharedB = sharedMem + AB_SMEM_STRIDE*BLOCK_ROWS; // 256x8

    /*
        for every block: 
        A moves 256 elements along row (y-direction)
        B moves 128 elements along col (x-direction)
        C moves 256 elements along row & 128 along col
     */
    const half* tileA = &A((by<<7), 0);
    const half* tileB = &B(0, (bx<<7));
    half* tileC = &C((by<<7), (bx<<7));

    half2 RA[4], RB[4], tmp_mul[4];

    // every thread response to 8x8 elements in C
    half RC[MMA_M][MMA_N]; // RC[8][8]
    memset(RC, 0, sizeof(RC));

    const size_t K_tiles = div_ceil(K, MMA_K*CHUNK_K);
    for(int iter_k=0; iter_k<K_tiles; ++iter_k) {
        /* LDG STS */
        ldgstsAOpt2(sharedA, tileA, 
                    tile_row_a, tile_col_a, 
                    smem_row_a, lda);
        ldgstsBOpt2(sharedB, tileB, 
                    tile_row_b, tile_col_b, 
                    smem_col_b, ldb);

        tileA += AB_SMEM_STRIDE;
        tileB += AB_SMEM_STRIDE;
        __syncthreads();

        #pragma unroll
        for(int iner_iter=0; iner_iter<CHUNK_K; ++iner_iter) {
            /*
               warp_row_c = 32*(0,1,2,3)
               warp_col_c = 64*(0,1)

               warp_tid_row = 8*(0,0,0,0,0,0,0,0| ... | 3,3,3,3,3,3,3,3)
               warp_tid_col = (0,1,2,3,4,5,6,7| ... | 0,1,2,3,4,5,6,7)
             */

            #pragma unroll
            for(int i=0; i<MMA_M; ++i) { // i=0,1,2,3

                //half2 RA[4], RB[4], tmp_mul[4];
                *((float4*)RA) = *(float4*)(&sharedA(warp_row_c+warp_tid_row+i+iner_iter*BLOCK_ROWS));

                for(int j=0; j<MMA_N; ++j) {
                    *((float4*)RB) = *(float4*)(&sharedB(warp_col_c+warp_tid_col+(j<<3)+iner_iter*BLOCK_COLS));

                    // half2 RA[4], RB[4];
                    // half RC[MMA_M][MMA_N]; // RC[8][8]
                    tmp_mul[0] = __hmul2(RA[0], RB[0]);
                    tmp_mul[1] = __hmul2(RA[1], RB[1]);
                    tmp_mul[2] = __hmul2(RA[2], RB[2]);
                    tmp_mul[3] = __hmul2(RA[3], RB[3]);

                    tmp_mul[0] = __hadd2(tmp_mul[0], tmp_mul[1]); 
                    tmp_mul[1] = __hadd2(tmp_mul[2], tmp_mul[3]); 

                    tmp_mul[0] = __hadd2(tmp_mul[0], tmp_mul[1]); 

                    RC[i][j] += tmp_mul[0].x + tmp_mul[0].y;
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for(int i=0; i<MMA_M; ++i) {
        /*
           warp_row_c = 32*(0,1,2,3)
           warp_col_c = 64*(0,1)

           warp_tid_row = 16*(0,0,0,0,0,0,0,0| ... | 3,3,3,3,3,3,3,3)
           warp_tid_col = 8*(0,1,2,3,4,5,6,7| ... | 0,1,2,3,4,5,6,7)
         */
        #pragma unroll
        for(int j=0; j<MMA_N; ++j) {
            tileC(warp_row_c+warp_tid_row+i, warp_col_c+warp_tid_col+j*8) = RC[i][j];
        }
    }
}

size_t initSimdWarpReorderOpt2() {
    int dev_id = 0;
    HGEMM_CHECK_CUDART_ERROR(cudaGetDevice(&dev_id));

    cudaDeviceProp dev_prop;
    HGEMM_CHECK_CUDART_ERROR(cudaGetDeviceProperties(&dev_prop, dev_id));

    size_t smem_max_size =
        std::max((BLOCK_ROWS + BLOCK_COLS) * AB_SMEM_STRIDE * sizeof(half), 
                  BLOCK_ROWS * BLOCK_COLS* sizeof(half));
    HLOG("smem_max_size: %.0f KBytes (%zu Bytes)", static_cast<double>(smem_max_size) / 1024, smem_max_size);

    HGEMM_CHECK_GT(dev_prop.sharedMemPerMultiprocessor, smem_max_size);
    HGEMM_CHECK_CUDART_ERROR(
        cudaFuncSetAttribute(simdWarpReorderKernelOpt2, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max_size));

    return smem_max_size;
}

void simdWarpReorderOpt2(half *A, half *B, half *C, size_t M, size_t N, size_t K) {
    size_t smem_max_size = initSimdWarpReorderOpt2();

    dim3 block(THREADS_PER_BLOCK);
    //#define BLOCK_ROWS 128
    //#define BLOCK_COLS 128
    dim3 grid(div_ceil(N, BLOCK_COLS), div_ceil(M, BLOCK_ROWS));

    simdWarpReorderKernelOpt2<<<grid, block, smem_max_size>>>(A, B, C, M, N, K);
}


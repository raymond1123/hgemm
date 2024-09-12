// Copyright 2023. All Rights Reserved.
// Author: Raymond
// Date: 16:27:28 on Fri, Jun 07, 2024
//
// Description: simd naive

#include "common.h"

#define THREADS_PER_BLOCK 256

#define BLOCK_STRIDE 2
#define BLOCK_ROW_TILES 16  // BLOCK_COLS / MMA_N
#define BLOCK_COL_TILES 16  // BLOCK_ROWS / MMA_M

#define BLOCK_ROWS 128
#define BLOCK_COLS 128
#define AB_SMEM_STRIDE 16

#define MMA_M 8
#define MMA_N 8
#define MMA_K 8
#define CHUNK_K 2

#define K_STAGE 2
#define SMEM_OFFSET ((BLOCK_ROWS + BLOCK_COLS)*AB_SMEM_STRIDE)

#define WARP_ROW_TILES 4
#define WARP_COL_TILES 8

#define tileA(i,j) tileA[(i)*(lda)+(j)]
#define tileB(i,j) tileB[(j)*(ldb)+(i)]
#define tileC(i,j) tileC[(i)*(ldc)+(j)]

#define sharedA(i, k) sharedA[(i)*MMA_K+(k)*SMEM_OFFSET]
#define sharedB(i, k) sharedB[(i)*MMA_K+(k)*SMEM_OFFSET]
#define sharedMem(i,j) sharedMem[(i)*(BLOCK_COLS)+(j)]

#define A(i,j) A[(i)*lda+(j)]
#define B(i,j) B[(i)+(j)*ldb]
#define C(i,j) C[(i)*ldc+(j)]

__global__ void simdWarpReorderKernelOpt7(const half *__restrict__ A, 
                                      const half *__restrict__ B, 
                                      half *__restrict__ C, 
                                      size_t M, size_t N, size_t K) {

    int lda=M, ldb=K, ldc=M;
    int tx = threadIdx.x;

    int warp_id = tx>>5;
    int lane_id = tx&31;

    /*
       every block has 8 warps: 4x2
       warp.shape = (4,2) ==> warp_row=0,1,2,3; warp_col=0,1
     */
    int warp_row = warp_id & 3, warp_col = warp_id >> 2;

    /* 
       tile_row_a = 0,0|1,1|...|127,127
       tile_col_a = 0,8|0,8|...|0,8
     */
    int tile_row_a = tx>>1, tile_col_a = (tx&1)<<3;

    /* 
       smem_row_a = 0,128,1,129,...,127,255
     */
    int smem_row_a = (tx&1)==0?(tx>>1):((tx>>1)+128);

    /* 
       tile_row_b = 0,8|0,8|...|0,8
       tile_col_b = 0,0|1,1|...|127,127
     */
    int tile_row_b = (tx&1)<<3, tile_col_b = tx>>1;

    /* 
       smem_col_b = 0,128,1,129,...,127,255
     */
    int smem_col_b = (tx&1)==0?(tx>>1):((tx>>1)+128);

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
    half* sharedB = sharedA + AB_SMEM_STRIDE*BLOCK_ROWS; // 256x8

    /* block swizzle */
    int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;

    const size_t block_tile_i = (bz&1)?by:(gridDim.y-by-1);
    const size_t block_tile_j = bz * gridDim.x + bx;

    const half* tileA = &A((block_tile_i<<7), 0);
    const half* tileB = &B(0, (block_tile_j<<7));
    half* tileC = &C((block_tile_i<<7), (block_tile_j<<7));

    half2 RA[K_STAGE][4], RB[K_STAGE][4], tmp_mul[4];
    float4 pref_A, pref_B;

    // every thread response to 16x8 elements in C
    half RC[MMA_M][MMA_N]; // RC[8][8]
    memset(RC, 0, sizeof(RC));

    /* first LDG STS */
    *((float4*)(&sharedA(smem_row_a, 0))) = *((float4*)(&tileA(tile_row_a, tile_col_a)));
    *((float4*)(&sharedB(smem_col_b, 0))) = *((float4*)(&tileB(tile_row_b, tile_col_b)));

    int iner_row = warp_row_c + warp_tid_row + (lane_id&7);
    int iner_col = warp_col_c+warp_tid_col;

    *((float4*)RA[0]) = *(float4*)(&sharedA(iner_row, 0));
    *((float4*)RB[0]) = *(float4*)(&sharedB(iner_col, 0));
    __syncthreads();

    const size_t K_tiles = div_ceil(K, MMA_K*CHUNK_K);
    for(int iter_k=0; iter_k<K_tiles; ++iter_k) {

        if (iter_k < K_tiles-K_STAGE) {
            tileA += AB_SMEM_STRIDE;
            tileB += AB_SMEM_STRIDE;

            pref_A = *((float4*)(&tileA(tile_row_a, tile_col_a)));
            pref_B = *((float4*)(&tileB(tile_row_b, tile_col_b)));
        }

        #pragma unroll
        for(int i=0; i<MMA_M; ++i) { // i=0,1,2,...,7
            int iner_row = warp_row_c + warp_tid_row + (((lane_id&7)+i+1)&7);

            if(i==MMA_M-1)
                *((float4*)RA[0]) = *(float4*)(&sharedA(BLOCK_ROWS+warp_row_c+warp_tid_row+(lane_id&7), iter_k%K_STAGE));
            else
                *((float4*)RA[(i+1)%K_STAGE]) = *(float4*)(&sharedA(iner_row, iter_k%K_STAGE));

            for(int j=0; j<MMA_N; ++j) {
                int iner_col = (i&1)?(warp_col_c+warp_tid_col+((MMA_N-j)<<3)):(warp_col_c+warp_tid_col+(j<<3));

                if(j==MMA_N-1)
                    *((float4*)RB[0]) = *(float4*)(&sharedB(BLOCK_COLS+warp_col_c+warp_tid_col, iter_k%K_STAGE));
                else
                    *((float4*)RB[(j+1)%K_STAGE]) = *(float4*)(&sharedB(iner_col, iter_k%K_STAGE));

                tmp_mul[0] = __hmul2(RA[i%K_STAGE][0], RB[j%K_STAGE][0]);
                tmp_mul[1] = __hmul2(RA[i%K_STAGE][1], RB[j%K_STAGE][1]);
                tmp_mul[2] = __hmul2(RA[i%K_STAGE][2], RB[j%K_STAGE][2]);
                tmp_mul[3] = __hmul2(RA[i%K_STAGE][3], RB[j%K_STAGE][3]);

                tmp_mul[0] = __hadd2(tmp_mul[0], tmp_mul[1]); 
                tmp_mul[1] = __hadd2(tmp_mul[2], tmp_mul[3]); 

                tmp_mul[0] = __hadd2(tmp_mul[0], tmp_mul[1]); 

                half tmp = tmp_mul[0].x + tmp_mul[0].y;

                if((i&1)==0)
                    RC[i][j] += tmp;
                else
                    RC[i][MMA_N-j-1] += tmp;
            }
        }

        #pragma unroll
        for(int i=0; i<MMA_M; ++i) { // i=0,1,2,...,7
            int iner_row = warp_row_c + warp_tid_row + (((lane_id&7)+i+1)&7);

            *((float4*)RA[(i+1)%K_STAGE]) = *(float4*)(&sharedA(BLOCK_ROWS+iner_row, iter_k%K_STAGE));

            for(int j=0; j<MMA_N; ++j) {
                int iner_col = (i&1)?(warp_col_c+warp_tid_col+((MMA_N-j)<<3)):(warp_col_c+warp_tid_col+(j<<3));
                *((float4*)RB[(j+1)%K_STAGE]) = *(float4*)(&sharedB(BLOCK_COLS+iner_col, iter_k%K_STAGE));

                tmp_mul[0] = __hmul2(RA[i%K_STAGE][0], RB[j%K_STAGE][0]);
                tmp_mul[1] = __hmul2(RA[i%K_STAGE][1], RB[j%K_STAGE][1]);
                tmp_mul[2] = __hmul2(RA[i%K_STAGE][2], RB[j%K_STAGE][2]);
                tmp_mul[3] = __hmul2(RA[i%K_STAGE][3], RB[j%K_STAGE][3]);

                tmp_mul[0] = __hadd2(tmp_mul[0], tmp_mul[1]); 
                tmp_mul[1] = __hadd2(tmp_mul[2], tmp_mul[3]); 

                tmp_mul[0] = __hadd2(tmp_mul[0], tmp_mul[1]); 

                half tmp = tmp_mul[0].x + tmp_mul[0].y;

                if((i&1)==0)
                    RC[i][j] += tmp;
                else
                    RC[i][MMA_N-j-1] += tmp;
            }
        }

        /* LDG STS */
        if (iter_k < K_tiles-K_STAGE) {

            *((float4*)(&sharedA(smem_row_a, (iter_k+1)%K_STAGE))) = pref_A;
            *((float4*)(&sharedB(smem_col_b, (iter_k+1)%K_STAGE))) = pref_B;

            int iner_row = warp_row_c + warp_tid_row + (lane_id&7);
            int iner_col = warp_col_c+warp_tid_col;

            *((float4*)RA[0]) = *(float4*)(&sharedA(iner_row, (iter_k+1)%K_STAGE));
            *((float4*)RB[0]) = *(float4*)(&sharedB(iner_col, (iter_k+1)%K_STAGE));

        }
        __syncthreads();
    }

    #pragma unroll
    for(int i=0; i<MMA_M; ++i) {
        int iner_row1 = ((lane_id&7)+i)&7;
        /* 
           ((lane_id&7)+0)&7 = (0,1,2,3, 4, 5, 6,  7)&7  ==> 0,1,2,3,4,5,6,7
           ((lane_id&7)+1)&7 = (1,2,3,4, 5, 6, 7,  8)&7  ==> 1,2,3,4,5,6,7,0
           ((lane_id&7)+2)&7 = (2,3,4,5, 6, 7, 8,  9)&7  ==> 2,3,4,5,6,7,0,1
           ((lane_id&7)+3)&7 = (3,4,5,6, 7, 8, 9, 10)&7  ==> 3,4,5,6,7,0,1,2
           ((lane_id&7)+4)&7 = (4,5,6,7, 8, 9, 10,11)&7  ==> 4,5,6,7,0,1,2,3
           ((lane_id&7)+5)&7 = (5,6,7,8, 9, 10,11,12)&7  ==> 5,6,7,0,1,2,3,4
           ((lane_id&7)+6)&7 = (6,7,8,9, 10,11,12,13)&7  ==> 6,7,0,1,2,3,4,5
           ((lane_id&7)+7)&7 = (7,8,9,10,11,12,13,14)&7  ==> 7,0,1,2,3,4,5,6
         */
        #pragma unroll
        for(int j=0; j<MMA_N; ++j) {
            /*
               warp_row_c = 32*(0,1,2,3)
               warp_col_c = 64*(0,1)

               warp_tid_row = 8*(0,0,0,0,0,0,0,0| ... | 3,3,3,3,3,3,3,3)
               warp_tid_col = (0,1,2,3,4,5,6,7| ... | 0,1,2,3,4,5,6,7)
             */

            sharedMem(warp_row_c+warp_tid_row+iner_row1, warp_col_c+warp_tid_col+(j<<3)) = RC[i][j];
        }
    }
    __syncthreads();

    /*
       warp_global_row_c = 0,0,...,0 | 1,1,...,1
       warp_global_col_c = 0,8,16,...,120 | 0,8,16,...,120
     */
    int warp_global_row_c = lane_id>>4;
    int warp_global_col_c = (lane_id&15)<<3;
    int row_warp = (warp_id<<4)+warp_global_row_c;

    #pragma unroll
    for(int i=0; i<8; ++i) {
        *(float4*)(&tileC(row_warp+i*2, warp_global_col_c)) = 
            *(float4*)(&sharedMem(row_warp+i*2, warp_global_col_c));
    }
}

size_t initSimdWarpReorderOpt7() {
    int dev_id = 0;
    HGEMM_CHECK_CUDART_ERROR(cudaGetDevice(&dev_id));

    cudaDeviceProp dev_prop;
    HGEMM_CHECK_CUDART_ERROR(cudaGetDeviceProperties(&dev_prop, dev_id));

    size_t smem_max_size = std::max((BLOCK_ROWS + BLOCK_COLS)*AB_SMEM_STRIDE*sizeof(half)*K_STAGE, 
                                    BLOCK_ROWS * BLOCK_COLS* sizeof(half));
    HLOG("smem_max_size: %.0f KBytes (%zu Bytes)", static_cast<double>(smem_max_size) / 1024, smem_max_size);

    HGEMM_CHECK_GT(dev_prop.sharedMemPerMultiprocessor, smem_max_size);
    HGEMM_CHECK_CUDART_ERROR(
        cudaFuncSetAttribute(simdWarpReorderKernelOpt7, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max_size));

    return smem_max_size;
}

void simdWarpReorderOpt7(half *A, half *B, half *C, size_t M, size_t N, size_t K) {
    size_t smem_max_size = initSimdWarpReorderOpt7();

    dim3 block(THREADS_PER_BLOCK);
    //#define BLOCK_ROWS 128
    //#define BLOCK_COLS 128
    //#define AB_SMEM_STRIDE 8
    dim3 grid(BLOCK_STRIDE, div_ceil(M, BLOCK_ROWS), div_ceil(N, BLOCK_COLS*BLOCK_STRIDE));

    simdWarpReorderKernelOpt7<<<grid, block, smem_max_size>>>(A, B, C, M, N, K);
}


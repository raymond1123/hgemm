// Copyright 2023. All Rights Reserved.
// Author: Raymond
// Date: 16:27:28 on Fri, Jun 07, 2024
//
// Description: simd warp reorder

#include "common.h"

#define THREADS_PER_BLOCK 256

#define BLOCK_ROWS 256
#define BLOCK_COLS 128
#define AB_SMEM_STRIDE 32

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16
#define CHUNK_K 2

#define LOAD_SMEM_STRIDE 64
#define WARP_ROW_TILES 4  // WARP_ROWS / LOAD_SMEM_STRIDE
#define WARP_COL_TILES 8  // WARP_COLS / LOAD_SMEM_STRIDE

#define tileA(i,j) tileA[(i)*(lda)+(j)]
#define tileB(i,j) tileB[(j)*(ldb)+(i)]
#define tileC(i,j) tileC[(i)*(ldc)+(j)]

#define sharedA(i,j) sharedA[(i)*(AB_SMEM_STRIDE)+(j)]
#define sharedB(i,j) sharedB[(i)*(AB_SMEM_STRIDE)+(j)]
#define sharedMem(i,j) sharedMem[(i)*(BLOCK_COLS)+(j)]

#define A(i,j) A[(i)*lda+(j)]
#define B(i,j) B[(i)+(j)*ldb]
#define C(i,j) C[(i)*ldc+(j)]

__device__ void ldgstsA(half* sharedA, const half* tileA, 
                        int row_a, int col_a, int lda) {
    /* 
       row_a = 0,0, 0, 0| 1,1, 1, 1|,...| 63,63,63,63
       col_a = 0,8,16,24| 0,8,16,24|,...| 0,8,16,24
     */
    #pragma unroll
    for(int i=0; i<4; ++i)
        *((float4*)(&sharedA(i*LOAD_SMEM_STRIDE+row_a, col_a))) = 
            *((float4*)(&tileA(i*LOAD_SMEM_STRIDE+row_a, col_a)));
}

__device__ void ldgstsB(half* sharedB, const half* tileB, 
                        int row_b, int col_b, int ldb) {
    /* 
       row_b = 0,8,16,24| 0,8,16,24|,...| 0,8,16,24
       col_b = 0,0, 0, 0| 1,1, 1, 1|,...| 63,63,63,63
     */
    #pragma unroll
    for(int i=0; i<2; ++i)
        *((float4*)(&sharedB(i*LOAD_SMEM_STRIDE+col_b, row_b))) = 
            *((float4*)(&tileB(row_b, i*LOAD_SMEM_STRIDE+col_b)));
}

__global__ void simdWarpReorderKernel(const half *__restrict__ A, 
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
       row_a = 0,0, 0, 0| 1,1, 1, 1|,...| 63,63,63,63
       col_a = 0,8,16,24| 0,8,16,24|,...| 0,8,16,24
     */
    int row_a = tx>>2, col_a = (tx&3)<<3;

    /* 
       row_b = 0,8,16,24| 0,8,16,24|,...| 0,8,16,24
       col_b = 0,0, 0, 0| 1,1, 1, 1|,...| 63,63,63,63
     */
    int row_b = (tx&3)<<3, col_b = tx>>2;

    /*
       warp_row_c = 64*(0,1,2,3)
       warp_col_c = 64*(0,1)
     */
    int warp_row_c = warp_row<<6;
    int warp_col_c = warp_col<<6;

    /*
       warp_tid_row = 16*(0,0,0,0,0,0,0,0| ... | 3,3,3,3,3,3,3,3)
       warp_tid_col = 8*(0,1,2,3,4,5,6,7| ... | 0,1,2,3,4,5,6,7)
     */
    int warp_tid_row = (lane_id>>3)<<4;
    int warp_tid_col = (lane_id&7)<<3;

    // shared memory for input matrix A & B
    extern __shared__ half sharedMem[];
    half* sharedA = sharedMem; // 256x32
    half* sharedB = sharedMem + AB_SMEM_STRIDE*BLOCK_ROWS; // 128x32

    /*
        for every block: 
        A moves 256 elements along row (y-direction)
        B moves 128 elements along col (x-direction)
        C moves 256 elements along row & 128 along col
     */
    const half* tileA = &A((by<<8), 0);
    const half* tileB = &B(0, (bx<<7));
    half* tileC = &C((by<<8), (bx<<7));

    half2 RA[8], RB[8], tmp_mul[8];

    // every thread response to 16x8 elements in C
    half RC[MMA_M][MMA_N]; // RC[16][8]
    memset(RC, 0, sizeof(RC));

    const size_t K_tiles = div_ceil(K, MMA_K*CHUNK_K);
    for(int iter_k=0; iter_k<K_tiles; ++iter_k) {
        /* LDG STS */
        ldgstsA(sharedA, tileA, row_a, col_a, lda);
        ldgstsB(sharedB, tileB, row_b, col_b, ldb);

        tileA += AB_SMEM_STRIDE;
        tileB += AB_SMEM_STRIDE;
        __syncthreads();

        #pragma unroll
        for(int iner_iter=0; iner_iter<CHUNK_K; ++iner_iter) {
            /*
               warp_row_c = 64*(0,1,2,3)
               warp_col_c = 64*(0,1)

               warp_tid_row = 16*(0,0,0,0,0,0,0,0| ... | 3,3,3,3,3,3,3,3)
               warp_tid_col = 8*(0,1,2,3,4,5,6,7| ... | 0,1,2,3,4,5,6,7)
             */

            for(int i=0; i<MMA_N; ++i) {

                float4* tmp_R = (float4*)(&sharedB(warp_col_c+warp_tid_col+i, iner_iter*(AB_SMEM_STRIDE/2)));

                *((float4*)RB) = *(tmp_R); 
                *((float4*)(RB+4)) = *(tmp_R+1);

                for(int j=0; j<MMA_M; ++j) {
                    tmp_R = (float4*)(&sharedA(warp_row_c+warp_tid_row+j, iner_iter*(AB_SMEM_STRIDE/2)));

                    *((float4*)RA) = *(tmp_R);
                    *((float4*)(RA+4)) = *(tmp_R+1);

                    // half2 RA[8], RB[8];
                    // half RC[MMA_M][MMA_N]; // RC[16][8]
                    tmp_mul[0] = __hmul2(RA[0], RB[0]);
                    tmp_mul[1] = __hmul2(RA[1], RB[1]);
                    tmp_mul[2] = __hmul2(RA[2], RB[2]);
                    tmp_mul[3] = __hmul2(RA[3], RB[3]);
                    tmp_mul[4] = __hmul2(RA[4], RB[4]);
                    tmp_mul[5] = __hmul2(RA[5], RB[5]);
                    tmp_mul[6] = __hmul2(RA[6], RB[6]);
                    tmp_mul[7] = __hmul2(RA[7], RB[7]);

                    tmp_mul[0] = __hadd2(tmp_mul[0], tmp_mul[1]); 
                    tmp_mul[1] = __hadd2(tmp_mul[2], tmp_mul[3]); 
                    tmp_mul[2] = __hadd2(tmp_mul[4], tmp_mul[5]); 
                    tmp_mul[3] = __hadd2(tmp_mul[6], tmp_mul[7]); 

                    tmp_mul[0] = __hadd2(tmp_mul[0], tmp_mul[1]); 
                    tmp_mul[1] = __hadd2(tmp_mul[2], tmp_mul[3]); 

                    tmp_mul[0] = __hadd2(tmp_mul[0], tmp_mul[1]); 

                    RC[j][i] += __float2half((__half2float(tmp_mul[0].x) + 
                                              __half2float(tmp_mul[0].y)));
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for(int i=0; i<MMA_M; ++i) {
        /*
           warp_row_c = 64*(0,1,2,3)
           warp_col_c = 64*(0,1)

           warp_tid_row = 16*(0,0,0,0,0,0,0,0| ... | 3,3,3,3,3,3,3,3)
           warp_tid_col = 8*(0,1,2,3,4,5,6,7| ... | 0,1,2,3,4,5,6,7)
         */
        *(float4*)(&tileC(warp_row_c+warp_tid_row+i, warp_col_c+warp_tid_col)) = *(float4*)(&RC[i][0]);
    }
}

size_t initSimdWarpReorder() {
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
        cudaFuncSetAttribute(simdWarpReorderKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max_size));

    return smem_max_size;
}

void simdWarpReorder(half *A, half *B, half *C, size_t M, size_t N, size_t K) {
    size_t smem_max_size = initSimdWarpReorder();

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(div_ceil(N, BLOCK_COLS), div_ceil(M, BLOCK_ROWS));

    simdWarpReorderKernel<<<grid, block, smem_max_size>>>(A, B, C, M, N, K);
}


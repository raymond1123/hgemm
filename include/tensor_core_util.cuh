#pragma once
#include "common.h"

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define BLOCK_ROWS 256
#define BLOCK_COLS 128

#define WARP_ROWS 64
#define WARP_COLS 64

#define BLOCK_ROW_WARPS 2  // BLOCK_COLS / WARP_COLS: 2 warps every row
#define BLOCK_COL_WARPS 4  // BLOCK_ROWS / WARP_ROWS: 2 warps every col

#define BLOCK_ROW_TILES 16  // BLOCK_COLS / MMA_N
#define BLOCK_COL_TILES 16  // BLOCK_ROWS / MMA_M

#define WARP_ROW_TILES 8  // WARP_COLS / MMA_N
#define WARP_COL_TILES 4  // WARP_ROWS / MMA_M

#define WARPS_PER_BLOCK 8      // BLOCK_ROW_WARPS * BLOCK_COL_WARPS
#define THREADS_PER_BLOCK 256  // WARP_SIZE * WARPS_PER_BLOCK

#define CHUNK_K 2  // 32 / MMA_K
#define AB_SMEM_STRIDE 32  // CHUNK_K * MMA_K

#define C_SMEM_STRIDE 128  // BLOCK_COLS
#define C_SMEM_OFFSET 64   // WARP_COLS

#define BLOCK_STRIDE 16
#define LDGSTS_CP_ROWS 8
#define PERMUTED_OFFSET 8
#define PERMUTED_COLS 4

#define A_WARP_TILE(i,j) (A_warp_tile+(i)*K+((j)<<3))
#define B_WARP_TILE(i,j) (B_warp_tile+(j)*K+((i)<<3))

#define SMEMA(i,j) (smemA+(i)*AB_SMEM_STRIDE+(j))
#define SMEMB(i,j) (smemB+(i)*AB_SMEM_STRIDE+(j))
#define SMEMC(i) (smemC+(i)*C_SMEM_STRIDE)

#define A(i,j) A[((i)*(MMA_M)+(j)*BLOCK_ROWS/WARPS_PER_BLOCK)*K]
#define B(i,j) B[((i)*(MMA_N)+(j)*BLOCK_COLS/WARPS_PER_BLOCK)*K]
#define C(i,j) C[(i)*MMA_M*N+(j)*MMA_N]

/* 
    block sizzle:
        index i is along col direction, j is along row direction
        block_tile_i is for the i-th tile(MMA level) rather than block level

        block shape=(256,128) 
        ==> for m16n8k16 tile, every block contains (256/m, 128/n)=(16,16) tiles
        ==> right shift 4 bits stands for multiply by 16 on both block_tile_i and block_tile_j
*/ 
__device__ __inline__ void swizzle(size_t* block_tile_i, size_t* block_tile_j) {
    *block_tile_i = (blockIdx.z & 1)?((gridDim.y-blockIdx.y-1)<<4):(blockIdx.y<<4);
    *block_tile_j = (blockIdx.z * gridDim.x + blockIdx.x)<<4;
}

/* calc hm16k8n16 */
void __device__ __inline__ hm16n8k16(int i, int j, 
                                    uint32_t RA[WARP_COL_TILES][4],
                                    uint32_t RB[WARP_ROW_TILES][2],
                                    uint32_t RC[WARP_COL_TILES][WARP_ROW_TILES][2]) {
    /* adopt Right Left Right Left style */
    size_t j_s = (i&1) ? (WARP_ROW_TILES - j - 1) : j;

    HMMA16816(RC[i][j_s][0], RC[i][j_s][1], 
              RA[i][0], RA[i][1], RA[i][2], RA[i][3], 
              RB[j_s][0], RB[j_s][1], 
              RC[i][j_s][0], RC[i][j_s][1]);
}

void __device__ __inline__ hm16n8k16(int i, int j, int reg_idx,
                                    uint32_t RA[2][WARP_COL_TILES][4],
                                    uint32_t RB[2][WARP_ROW_TILES][2],
                                    uint32_t RC[WARP_COL_TILES][WARP_ROW_TILES][2]) {
    /* adopt Right Left Right Left style */
    size_t j_s = (i&1) ? (WARP_ROW_TILES - j - 1) : j;

    HMMA16816(RC[i][j_s][0], RC[i][j_s][1], 
              RA[reg_idx][i][0], RA[reg_idx][i][1], 
              RA[reg_idx][i][2], RA[reg_idx][i][3], 
              RB[reg_idx][j_s][0], RB[reg_idx][j_s][1], 
              RC[i][j_s][0], RC[i][j_s][1]);
}

/* ============== base version ============= */
/* Matrix A from HBM to SRAM */
__device__ void __inline__ ldgstsA_base(const size_t warp_id, const size_t lane_id, 
                                        const half* A_warp_ptr, 
                                        size_t tile_k, size_t K, half* smemA) {

    const half *A_warp_tile = A_warp_ptr+tile_k*MMA_K;

    /* 
        every single loop copy 8 rows and 4 cols ==> 
        lane_row = lane_id>>2 =[0,0,0,0, 1,1,1,1, ..., 7,7,7,7]
        lane_col = lane_id&3 = [0,1,2,3, 0,1,2,3, ..., 0,1,2,3]
     */
    int lane_row = lane_id>>2, lane_col=lane_id&3; 
    float4* A_lane_ptr = (float4*)(A_WARP_TILE(lane_row, lane_col));

    /* 
        every warp copy 32 rows from HBM to SRAM ==> warp_id<<5
        every time copy float4=8*half ==> col_A = (lane_id&3)<<3
     */
    size_t row_A = (warp_id<<5)+lane_row;
    size_t col_A = lane_col<<3;

    #pragma unroll
    for (size_t i = 0; i < 4; ++i) {
        // load current data
        *(float4*)(SMEMA(row_A, col_A)) = *A_lane_ptr;
        A_lane_ptr = (float4*)((half *)A_lane_ptr + LDGSTS_CP_ROWS*K);

        // prepare the next LDGSTS_CP_ROWS=8 rows
        row_A += LDGSTS_CP_ROWS;
    }
}

/* Matrix B from HBM to SRAM */
void __device__ __inline__ ldgstsB_base(const size_t warp_id, const size_t lane_id, const half* B_warp_ptr, 
                        size_t tile_k, size_t K, half* smemB) {

    const half *B_warp_tile = B_warp_ptr+tile_k*MMA_K;

    /* 
        every single loop copy 4 rows and 8 cols ==> 
        lane_row = lane_id&3 = [0,1,2,3, 0,1,2,3, ..., 0,1,2,3]
        lane_col = lane_id>>2 =[0,0,0,0, 1,1,1,1, ..., 7,7,7,7]
     */
    int lane_row = lane_id&3, lane_col=lane_id>>2; 
    float4* B_lane_ptr = (float4*)(B_WARP_TILE(lane_row, lane_col));

    /* 
        every warp copy 16 rows from HBM to SRAM ==> warp_id<<4
        every time copy float4=8*half ==> row_B = (lane_id&3)<<3
        Note: here is a transpose!
     */
    size_t row_B = (warp_id<<4)+lane_col;
    size_t col_B = lane_row<<3;

    #pragma unroll
    for (size_t i = 0; i < 2; ++i) {
        // load current data
        *(float4*)(SMEMB(row_B, col_B)) = *B_lane_ptr;

        // prepare the next 8 rows 
        B_lane_ptr = (float4*)((half *)B_lane_ptr + LDGSTS_CP_ROWS* K);
        row_B += LDGSTS_CP_ROWS;
    }
}

/* 
    load Matrix A data from SRAM to Register:
    every block has 2x4 warps, which is 2 rows and 4 cols ==>
    warp_row = warp_id>>1 = [0,1]
    every warp can be split into 4 rows and 8 cols of mma, contains 64x64 half elements
    every mma is m16n8k16
*/
void __device__ __inline__ ldsA_base(int i, int k_step, 
                                const size_t warp_id, const size_t lane_id,
                                half* smemA, uint32_t RA[WARP_COL_TILES][4]) {

    /* 
        every block is splited into 2x4 warps 
        every warp fetch 64x16 elements of Matrix A
    */
    size_t warp_row = (warp_id>>1)*WARP_ROWS + i*MMA_M;
    size_t warp_col = k_step*MMA_K;

    /* 
        lane_id&15 = [0,1,...,15, 0,1,...,15]
        lane_id>>4 = [0,1,2,3, ..., 0,1,2,3]
    */
    size_t lane_row = warp_row + (lane_id&15);
    size_t lane_col = warp_col + ((lane_id>>4)<<3); 

    //#define SMEMA(i,j) (smemA+(i)*AB_SMEM_STRIDE+(j))
    uint32_t A_smem_lane_addr =
        __cvta_generic_to_shared(SMEMA(lane_row, lane_col));

    LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], A_smem_lane_addr);
}

/* load Matrix B data from SRAM to Register */
void __device__ __inline__ ldsB_base(int j, int k_step, 
                                const size_t warp_id, const size_t lane_id,
                                half* smemB, uint32_t RB[WARP_COL_TILES][2]) {
    /* 
        every block is splited into 2x4 warps 
        every warp fetch 16x64 elements of Matrix B
    */
    size_t warp_row = (warp_id&1)*WARP_COLS + j*MMA_N;
    size_t warp_col = k_step*MMA_K;

    /* 
        for Matrix B, we only cares about the first 16 threads 
        lane_id&7  = [0,1,...,7, 0,1,...,7]
        lane_id>>3 = [0,...0,    1,...,1]
    */
    size_t lane_row = warp_row + (lane_id&7);
    size_t lane_col = warp_col + (((lane_id>>3)&1)<<3); 

    uint32_t B_smem_lane_addr =
        __cvta_generic_to_shared(SMEMB(lane_row, lane_col));

    LDMATRIX_X2(RB[j][0], RB[j][1], B_smem_lane_addr);
}

/* store Matrix C result from Register RC to SRAM */
void __device__ __inline__ stsC_base(int i, int j, const size_t warp_id, const size_t lane_id, 
                                     half* smem_warp_tile_ptr, 
                                     uint32_t RC[WARP_COL_TILES][WARP_ROW_TILES][2]) {

    /* 
        lane_id>>2 = [0,0,0,0, 1,1,1,1, ..., 7,7,7,7]
        lane_id&3  = [0,1,2,3, 0,1,2,3, ..., 0,1,2,3]
    */
    int row = i*MMA_M + (lane_id>>2);
    int col = j*MMA_N;

    half *lane_ptr0 = smem_warp_tile_ptr +
                      row*C_SMEM_STRIDE + col +
                      (lane_id&3) * sizeof(uint32_t) / sizeof(half);

    half *lane_ptr1 = smem_warp_tile_ptr + 
                      (row+8)*C_SMEM_STRIDE + col +
                      (lane_id&3) * sizeof(uint32_t) / sizeof(half);

    *((uint32_t *)(lane_ptr0)) = RC[i][j][0];
    *((uint32_t *)(lane_ptr1)) = RC[i][j][1];
}

/* store Matrix C result from SRAM to HBM */
void __device__ __inline__ ldsC_base(int i, const int N, 
                                     const size_t lane_id, 
                                     const half* src_gmem_warp_stream_ptr,
                                     const half* smem_warp_stream_ptr) {
    // tid=0, 1, ...,15 ==> lane_id/16=0
    // tid=16,17,...,31 ==> lane_id/16=1
    *((float4*)(src_gmem_warp_stream_ptr + (i*2+(lane_id>>4))*N)+(lane_id&15)) =
        *((float4*)(smem_warp_stream_ptr + (i*2+(lane_id>>4))*C_SMEM_STRIDE)+(lane_id&15));
}

/* ============== permutation version ============= */
/* 
    in permute version, only col_A different from ldgstsA_base, 
    considering bank conflict
*/
void __device__ __inline__ ldgstsA_permute(const size_t warp_id, const size_t lane_id, const half* A_warp_ptr, 
                                size_t tile_k, size_t K, half* smemA) {


    const half *A_warp_tile = A_warp_ptr+tile_k*MMA_K;

    int lane_row = lane_id>>2, lane_col=lane_id&3; 
    float4* A_lane_ptr = (float4*)(A_WARP_TILE(lane_row, lane_col));

    /* 
        ((row_A&7)>>1) = [0,0, 1,1, 2,2, 3,3]
        this is actually col offset
        the final '&3' is to reverse the residual to the beginning
    */
    size_t row_A = (warp_id<<5)+lane_row;
    size_t col_A = ((lane_col+((row_A&7)>>1))&3)<<3;

    #pragma unroll
    for (size_t i = 0; i < 4; ++i) {
        *(float4*)(SMEMA(row_A, col_A)) = *A_lane_ptr;

        A_lane_ptr = (float4*)((half *)A_lane_ptr + LDGSTS_CP_ROWS*K);
        row_A += LDGSTS_CP_ROWS;
    }
}

/*
    in permute version, only col_B different from ldgstsB_base, 
    considering bank conflict
*/
void __device__ __inline__ ldgstsB_permute(const size_t warp_id, const size_t lane_id, const half* B_warp_ptr, 
                        size_t tile_k, size_t K, half* smemB) {

    const half *B_warp_tile = B_warp_ptr+tile_k*MMA_K;

    int lane_row = lane_id&3, lane_col=lane_id>>2; 
    float4* B_lane_ptr = (float4*)(B_WARP_TILE(lane_row, lane_col));

    /* 
        ((row_B&7)>>1) = [0,0, 1,1, 2,2, 3,3]
        the same as row_A, this is actually col offset
        the final '&3' is to take the residual to the beginning position
    */
    size_t row_B = (warp_id<<4)+lane_col;
    size_t col_B = ((lane_row+((row_B&7)>>1))&3)<<3; 

    #pragma unroll
    for (size_t i = 0; i < 2; ++i) {
        // load current data
        *(float4*)(SMEMB(row_B, col_B)) = *B_lane_ptr;

        // prepare the next 8 rows 
        B_lane_ptr = (float4*)((half *)B_lane_ptr + LDGSTS_CP_ROWS* K);
        row_B += LDGSTS_CP_ROWS;
    }
}

/* load Matrix A data from SRAM to Register:
   in permute version, 
   only lane_col is different, considering bank conflict */
void __device__ __inline__ ldsA_permute(int i, int k_step, 
                                const size_t warp_id, const size_t lane_id,
                                half* smemA, uint32_t RA[WARP_COL_TILES][4]) {

    size_t warp_row = (warp_id>>1)*WARP_ROWS + i*MMA_M;
    size_t warp_col = k_step*MMA_K;

    /*  
        (((lane_id&15)&7)>>1)<<3;
        ((lane_id&15)&7) = ([0,1,...,7,      0,1,...,7,       0,1,...,7,       0,1,...,7])>>1
                         = [0,0,1,1,...,3,3, 0,0,1,1,...,3,3, 0,0,1,1,...,3,3, 0,0,1,1,...,3,3]
        the final '&31' is to take the residual to the beginning position
    */
    size_t lane_row = warp_row + (lane_id&15);
    size_t lane_col = warp_col + ((lane_id>>4)<<3) + ((((lane_id&15)&7)>>1)<<3);
    lane_col &= 31; 

    uint32_t A_smem_lane_addr =
        __cvta_generic_to_shared(SMEMA(lane_row, lane_col));

    LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], A_smem_lane_addr);
}

/* load Matrix B data from SRAM to Register:
    in permute version, 
    only lane_col is different, considering bank conflict */
void __device__ __inline__ ldsB_permute(int j, int k_step, 
                                const size_t warp_id, const size_t lane_id,
                                half* smemB, uint32_t RB[WARP_ROW_TILES][2]) {

    size_t warp_row = (warp_id&1)*WARP_COLS + j*MMA_N;
    size_t warp_col = k_step*MMA_K;

    size_t lane_row = warp_row + (lane_id&7);
    size_t lane_col = warp_col + (((lane_id>>3)&1)<<3) + ((((lane_id&7)&7)>>1)<<3);
    lane_col &= 31; 

    uint32_t B_smem_lane_addr =
        __cvta_generic_to_shared(SMEMB(lane_row, lane_col));

    LDMATRIX_X2(RB[j][0], RB[j][1], B_smem_lane_addr);
}

/* store Matrix C result from Register RC to SRAM */
void __device__ __inline__ stsC_permute(int i, int j, const size_t warp_id, const size_t lane_id, 
                                        half* smem_warp_tile_ptr, 
                                        uint32_t RC[WARP_COL_TILES][WARP_ROW_TILES][2]) {

    int row = i*MMA_M + (lane_id>>2);
    int col = j*MMA_N;
    int warp_offset = (warp_id&1)*C_SMEM_OFFSET;

    /* 
        (lane_id>>2)&7 = [0,0,0,0, 1,1,1,1, ..., 7,7,7,7]
        this is offset of permute of cols
    */
    half *lane_ptr0 = smem_warp_tile_ptr + row*C_SMEM_STRIDE + 
                      ((warp_offset + col +
                      (lane_id&3) * sizeof(uint32_t) / sizeof(half) + 
                      (((lane_id>>2)&7)<<3))&127);

    half *lane_ptr1 = smem_warp_tile_ptr + (row+8) * C_SMEM_STRIDE +
                      ((warp_offset + col +
                      (lane_id&3) * sizeof(uint32_t) / sizeof(half) + 
                      ((((lane_id>>2)+8)&7)<<3))&127);

    *((uint32_t *)(lane_ptr0)) = RC[i][j][0];
    *((uint32_t *)(lane_ptr1)) = RC[i][j][1];
}

/* store Matrix C result from SRAM to HBM */
void __device__ __inline__ ldsC_permute(int i, const int N, 
                                        const size_t lane_id, 
                                        const half* src_gmem_warp_stream_ptr,
                                        const half* smem_warp_stream_ptr) {
    // tid=0, 1, ...,15 ==> lane_id/16=0
    // tid=16,17,...,31 ==> lane_id/16=1

    /*  
        lane_id>>4 = [0,0,0,0, 1,1,1,1, ..., 7,7,7,7]
        (lane_id>>4)&7 = [0,0,0,0, 1,1,1,1, ..., 7,7,7,7]
    */
    *((float4*)(src_gmem_warp_stream_ptr + (i*2+(lane_id>>4))*N)+(lane_id&15)) =
        *((float4*)(smem_warp_stream_ptr + (i*2+(lane_id>>4))*C_SMEM_STRIDE)+
              (((lane_id&15)+(i*2+(lane_id>>4)&7))&15));

}

/* ============== async version ============= */
/* 
   only ldgsts A&B are different, 
   other operations are the same as permute version
*/
void __device__ __inline__ ldgstsA_async(const size_t warp_id, const size_t lane_id, 
                                         const half* A_warp_ptr, 
                                         size_t tile_k, size_t K, half* smemA) {

    const half *A_warp_tile = A_warp_ptr+tile_k*MMA_K;

    int lane_row = lane_id>>2, lane_col=lane_id&3; 
    float4* A_lane_ptr = (float4*)(A_WARP_TILE(lane_row, lane_col));

    size_t row_A = (warp_id<<5)+lane_row;
    size_t col_A = (lane_col+((row_A&7)>>1))&3;

    #pragma unroll
    for (size_t i = 0; i < 4; ++i) {
        // load current data
        uint32_t A_smem_lane_addr = __cvta_generic_to_shared(SMEMA(row_A,0)) + col_A*sizeof(float4);
        CP_ASYNC_CG(A_smem_lane_addr, A_lane_ptr, sizeof(float4));

        // prepare the next 8 rows 
        A_lane_ptr = (float4*)((half *)A_lane_ptr + LDGSTS_CP_ROWS*K);
        row_A += LDGSTS_CP_ROWS;
    }
}

void __device__ __inline__ ldgstsB_async(const size_t warp_id, const size_t lane_id, 
                                         const half* B_warp_ptr, 
                                         size_t tile_k, size_t K, half* smemB) {

    const half *B_warp_tile = B_warp_ptr+tile_k*MMA_K;

    int lane_row = lane_id&3, lane_col=lane_id>>2; 
    float4* B_lane_ptr = (float4*)(B_WARP_TILE(lane_row, lane_col));

    size_t row_B = (warp_id<<4)+lane_col;
    size_t col_B = (lane_row+((row_B&7)>>1))&3;

    #pragma unroll
    for (size_t i = 0; i < 2; ++i) {
        // load current data
        uint32_t B_smem_lane_addr = __cvta_generic_to_shared(SMEMB(row_B,0)) + col_B*sizeof(float4);
        CP_ASYNC_CG(B_smem_lane_addr, B_lane_ptr, sizeof(float4));

        // prepare the next 8 rows 
        B_lane_ptr = (float4*)((half *)B_lane_ptr + LDGSTS_CP_ROWS* K);
        row_B += LDGSTS_CP_ROWS;
    }
}

/* ============== stage_2 version ============= */
void __device__ __inline__ ldgstsA_stage(const size_t warp_id, const size_t lane_id, 
                                         const half* A_warp_ptr, 
                                         size_t tile_k, size_t K, half* smemA,
                                         const size_t smem_store_off) {

    const half *A_warp_tile = A_warp_ptr+tile_k*MMA_K;

    int lane_row = lane_id>>2, lane_col=lane_id&3; 
    float4* A_lane_ptr = (float4*)(A_WARP_TILE(lane_row, lane_col));

    size_t row_A = smem_store_off+(warp_id<<5)+lane_row;
    size_t col_A = (lane_col+((row_A&7)>>1))&3;

    #pragma unroll
    for (size_t i = 0; i < 4; ++i) {
        // load current data
        uint32_t A_smem_lane_addr = __cvta_generic_to_shared(SMEMA(row_A,0)) + col_A*sizeof(float4);
        CP_ASYNC_CG(A_smem_lane_addr, A_lane_ptr, sizeof(float4));

        // prepare the next 8 rows 
        A_lane_ptr = (float4*)((half *)A_lane_ptr + LDGSTS_CP_ROWS*K);
        row_A += LDGSTS_CP_ROWS;
    }
}

void __device__ __inline__ ldgstsB_stage(const size_t warp_id, const size_t lane_id, 
                                         const half* B_warp_ptr, 
                                         size_t tile_k, size_t K, half* smemB,
                                         const size_t smem_store_off) {

    const half *B_warp_tile = B_warp_ptr+tile_k*MMA_K;

    int lane_row = lane_id&3, lane_col=lane_id>>2; 
    float4* B_lane_ptr = (float4*)(B_WARP_TILE(lane_row, lane_col));

    size_t row_B = smem_store_off+(warp_id<<4)+lane_col;
    size_t col_B = (lane_row+((row_B&7)>>1))&3;

    #pragma unroll
    for (size_t i = 0; i < 2; ++i) {
        // load current data
        uint32_t B_smem_lane_addr = __cvta_generic_to_shared(SMEMB(row_B,0)) + col_B*sizeof(float4);
        CP_ASYNC_CG(B_smem_lane_addr, B_lane_ptr, sizeof(float4));

        // prepare the next 8 rows 
        B_lane_ptr = (float4*)((half *)B_lane_ptr + LDGSTS_CP_ROWS* K);
        row_B += LDGSTS_CP_ROWS;
    }
}

/* load Matrix A data from SRAM to Register */
void __device__ __inline__ ldsA_stage2(int i, int k_step, 
                                       size_t reg_store_idx, size_t smem_load_off,
                                       const size_t warp_id, const size_t lane_id,
                                       half* smemA, uint32_t RA[2][WARP_COL_TILES][4]) {

    size_t warp_row = smem_load_off + (warp_id>>1)*WARP_ROWS + i*MMA_M;
    size_t warp_col = k_step*MMA_K;

    size_t lane_row = warp_row + (lane_id&15);
    size_t lane_col = warp_col + ((lane_id>>4)<<3) + ((((lane_id&15)&7)>>1)<<3);
    lane_col &= 31; 

    uint32_t A_smem_lane_addr = __cvta_generic_to_shared(SMEMA(lane_row, lane_col));

    LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], 
                RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], 
                A_smem_lane_addr);
}

/* load Matrix B data from SRAM to Register */
void __device__ __inline__ ldsB_stage2(int j, int k_step, 
                                       size_t reg_store_idx, size_t smem_load_off,
                                       const size_t warp_id, const size_t lane_id,
                                       half* smemB, uint32_t RB[2][WARP_ROW_TILES][2]) {

    size_t warp_row = smem_load_off + (warp_id&1)*WARP_COLS + j*MMA_N;
    size_t warp_col = k_step*MMA_K;

    size_t lane_row = warp_row + (lane_id&7);
    size_t lane_col = warp_col + (((lane_id>>3)&1)<<3) + ((((lane_id&7)&7)>>1)<<3);
    lane_col &= 31; 

    uint32_t B_smem_lane_addr = __cvta_generic_to_shared(SMEMB(lane_row, lane_col));

    LDMATRIX_X2(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], B_smem_lane_addr);
}

/* ============== stage_3 version ============= */
/* load Matrix A data from SRAM to Register */
void __device__ __inline__ ldsA_stage3(int i, int k_step, 
                                       size_t reg_store_idx, size_t smem_load_off,
                                       const size_t warp_id, const size_t lane_id,
                                       half* smemA, uint32_t RA[2][WARP_COL_TILES][4]) {

    size_t warp_row = smem_load_off + (warp_id>>1)*WARP_ROWS + i*MMA_M;
    size_t warp_col = ((k_step+1)&1)*MMA_K;

    size_t lane_row = warp_row + (lane_id&15);
    size_t lane_col = warp_col + ((lane_id>>4)<<3) + ((((lane_id&15)&7)>>1)<<3);
    lane_col &= 31; 

    uint32_t A_smem_lane_addr = __cvta_generic_to_shared(SMEMA(lane_row, lane_col));

    LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], 
                RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], 
                A_smem_lane_addr);
}

/* load Matrix B data from SRAM to Register */
void __device__ __inline__ ldsB_stage3(int j, int k_step, 
                                       size_t reg_store_idx, size_t smem_load_off,
                                       const size_t warp_id, const size_t lane_id,
                                       half* smemB, uint32_t RB[2][WARP_ROW_TILES][2]) {

    size_t warp_row = smem_load_off + (warp_id&1)*WARP_COLS + j*MMA_N;
    size_t warp_col = ((k_step+1)&1)*MMA_K;

    size_t lane_row = warp_row + (lane_id&7);
    size_t lane_col = warp_col + (((lane_id>>3)&1)<<3) + ((((lane_id&7)&7)>>1)<<3);
    lane_col &= 31; 

    uint32_t B_smem_lane_addr = __cvta_generic_to_shared(SMEMB(lane_row, lane_col));

    LDMATRIX_X2(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], B_smem_lane_addr);
}




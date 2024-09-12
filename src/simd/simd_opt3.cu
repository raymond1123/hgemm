// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:02:28 on Tue, Feb 28, 2023
//
// Description: mma base hgemm

#include "common.h"

#define THREADS_PER_BLOCK 32 // WARP_SIZE * WARPS_PER_BLOCK
#define SMEM_SIZE 32
#define LOAD_NUM 8 // float4

#define A(i,j) A[(j)+(i)*lda] // row major
#define B(i,j) B[(i)+(j)*ldb] // col major
#define C(i,j) C[(j)+(i)*ldc] // row major

#define tileA(i,j) tileA[(j) + (i)*(lda/LOAD_NUM)] // row major
#define tileB(i,j) tileB[(i) + (j)*(ldb/LOAD_NUM)] // col major

#define sharedA(i,j) sharedA[(j)+(i)*SMEM_SIZE]
#define sharedB(i,j) sharedB[(j)+(i)*SMEM_SIZE]

__device__ void Opt3calc(const float4* loadA, 
                     const float4* loadB,
                     float* tmp) {

    half2 elem_a1 = *((half2*)loadA + 0);
    half2 elem_b1 = *((half2*)loadB + 0);
    half2 elem_a2 = *((half2*)loadA + 1);
    half2 elem_b2 = *((half2*)loadB + 1);
    half2 elem_a3 = *((half2*)loadA + 2);
    half2 elem_b3 = *((half2*)loadB + 2);
    half2 elem_a4 = *((half2*)loadA + 3);
    half2 elem_b4 = *((half2*)loadB + 3);

    half2 tmp1 = __hmul2(elem_a1, elem_b1);
    half2 tmp2 = __hmul2(elem_a2, elem_b2);
    half2 tmp3 = __hmul2(elem_a3, elem_b3);
    half2 tmp4 = __hmul2(elem_a4, elem_b4);

    half2 add_tmp1 = __hadd2(tmp1, tmp2);
    half2 add_tmp2 = __hadd2(tmp3, tmp4);
    half2 final_tmp3 = __hadd2(add_tmp1, add_tmp2);

    *tmp += __half2float(final_tmp3.x)+__half2float(final_tmp3.y);
}

__global__ void simdOpt3Kernel(const half *__restrict__ A, 
                              const half *__restrict__ B, 
                              half *__restrict__ C, 
                              size_t M, size_t N, size_t K) {

    // actually define two 32x32 shared mem for tile A and tile B
    extern __shared__ float4 sharedMem[];
    float4* sharedA = sharedMem;
    float4* sharedB = sharedMem + SMEM_SIZE*SMEM_SIZE;

    int lda=K, ldb=K, ldc=N;
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;

    float4* tileA = (float4*)(&A((by<<5), 0));
    float4* tileB = (float4*)(&B(0, (bx<<5)));
    C = &C((by<<5), (bx<<5));

    float tmp = 0.0;
    // in both smem ld or st, there should be no bankconflict at all.
    size_t iters = K/(SMEM_SIZE*LOAD_NUM);
    #pragma unroll
    for(size_t k=0; k<iters; ++k) {
        sharedA(ty, tx) = tileA(ty, tx);
        sharedB(ty, tx) = tileB(ty, tx);

        tileA += SMEM_SIZE;
        tileB += SMEM_SIZE;
        __syncthreads(); // wait for all threads in one block

        #pragma unroll
        for(size_t i=0; i<SMEM_SIZE; ++i) {
            float4* loadA = &(sharedA(ty,i));
            float4* loadB = &(sharedB(i,tx));

            Opt3calc(loadA, loadB, &tmp);

        }
        __syncthreads(); // wait for all threads in one block
    }

    C(ty,tx) = __float2half(tmp);
}

size_t initSimdOpt3() {
    int dev_id = 0;
    HGEMM_CHECK_CUDART_ERROR(cudaGetDevice(&dev_id));

    cudaDeviceProp dev_prop;
    HGEMM_CHECK_CUDART_ERROR(cudaGetDeviceProperties(&dev_prop, dev_id));

    size_t smem_max_size = 2*SMEM_SIZE*SMEM_SIZE*sizeof(float4);
    HLOG("smem_max_size: %.0f KBytes (%zu Bytes)", static_cast<double>(smem_max_size) / 1024, smem_max_size);

    HGEMM_CHECK_GT(dev_prop.sharedMemPerMultiprocessor, smem_max_size);
    HGEMM_CHECK_CUDART_ERROR(
        cudaFuncSetAttribute(simdOpt3Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max_size));

    return smem_max_size;
}

void simdOpt3(half *A, half *B, half *C, size_t M, size_t N, size_t K) {
    static size_t smem_max_size = initSimdOpt3();

    dim3 block(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 grid(div_ceil(M, THREADS_PER_BLOCK), div_ceil(N, THREADS_PER_BLOCK));

    simdOpt3Kernel<<<grid, block, smem_max_size>>>(A, B, C, M, N, K);
}


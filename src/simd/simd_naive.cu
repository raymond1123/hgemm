// Copyright 2023. All Rights Reserved.
// Author: Raymond
// Date: 16:27:28 on Fri, Jun 07, 2024
//
// Description: simd naive

#include "common.h"

#define THREADS_PER_BLOCK 32
#define A(i,j) A[(j) + (i)*lda]
#define B(i,j) B[(i) + (j)*ldb]
#define C(i,j) C[(j) + (i)*ldc]

__global__ void simdNaiveKernel(const half *__restrict__ A, 
                              const half *__restrict__ B, 
                              half *__restrict__ C, 
                              size_t M, size_t N, size_t K) {

    int lda=K, ldb=K, ldc=N;
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;

    A = &A((by<<5), 0);
    B = &B(0, (bx<<5));
    C = &C((by<<5), (bx<<5));

    //half tmp = 0.;
    float tmp = 0.;
    for(int k=0; k<K; ++k) {
        tmp += __half2float(A(ty,k))*__half2float(B(k,tx));
        //tmp += A(ty,k)*B(k,tx);
    }

    //C(ty,tx) = tmp;
    C(ty,tx) = __float2half(tmp);
}

void simdNaive(half *A, half *B, half *C, size_t M, size_t N, size_t K) {

    dim3 block(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 grid(div_ceil(M, THREADS_PER_BLOCK), div_ceil(N, THREADS_PER_BLOCK));

    simdNaiveKernel<<<grid, block>>>(A, B, C, M, N, K);
}


#ifndef __SIMD_UTIL_H
#define __SIMD_UTIL_H

#define THREADS_PER_BLOCK 32

#define A(i,j) A[(j) + (i)*lda] // row-major
#define B(i,j) B[(i) + (j)*ldb] // col-major
#define C(i,j) C[(j) + (i)*ldc] // row-major

#endif
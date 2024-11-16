# Introduction

基于木子知  [hgemm 代码](https://github.com/Bruce-Lee-LY/cuda_hgemm) ，对其进行了重构，使得整体流程更加清晰

对知乎上木子知[Tensor Core-CUDA HGEMM优化进阶](https://zhuanlan.zhihu.com/p/639297098)相关代码进行详尽分析

分析过程都是自己对理解，如果有误请提 issue



## mma

* [mma_base 分析](docs/mma/mma_base/mma_base.md)
* [mma_permute 分析](docs/mma/mma_permute/mma_permuted.md)
* [mma_async 分析](docs/mma/mma_async/mma_async.md)
* [mma_async_stage2 分析](docs/mma/mma_async_stage2/mma_async_stage2.md)
* [mma_async_stage3 分析](docs/mma/mma_async_stage3/mma_async_stage3.md)
* [mma_async_stage4 分析](docs/mma/mma_async_stage4/mma_async_stage4.md)



## TODO List

- [x] mma_base

- [x] mma_permuted

- [x] mma_async

- [x] mma_async_stage2

- [x] mma_async_stage3

- [x] mma_async_stage4

  


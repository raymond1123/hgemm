[toc]

## mma_async_stage4

相对于 mma_async_stage3 版本，mma_async_stage4 又增加了一组 global memory 异步拷贝到 shared memory。




### pipline

Figure-1 mma_async_stage3 的 pipeline

<img src="imgs/pipeline.png" alt="pipeline" style="zoom:80%;" />

<center> Figure-1 <center>

* mma_async_stage3 有3组 global memory 的异步拷贝

  ```c++
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
  ```

  * **<font color='red'> 注意这里的 CP_ASYNC_WAIT_GROUP(2); 即三组 global memory 异步拷贝</font>**

  

### Performance

测试的矩阵尺寸为 M=N=K=4096

Figure-2 是 mma_async_stage4 版本与 mma_async_stage3 版本的 ncu 性能指标对比

<img src="async3-vs-async4.png" alt="async3-vs-async4" style="zoom:80%;" />

<center> Figure-2 <center>

* 耗时1.42 ms，相对于 mma_async_stage3 性能提升已经接近极限




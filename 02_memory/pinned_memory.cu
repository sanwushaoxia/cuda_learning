// pinned_memory.cu — 页锁定内存(page-locked / pinned)加速传输
//
// 学习点:
//   1. malloc 的内存是可换页的,DMA 引擎传输时要先"锁定"到物理页,多一次中转
//   2. cudaMallocHost 分配页锁定内存,传输带宽通常提升 1.5~2 倍
//   3. cudaHostRegister 可以把已分配的大内存"补注册"为页锁定
//   4. 页锁定内存会减少系统可用物理页,不要滥用
//
// 编译运行: make run T=pinned_memory
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); exit(1); } \
} while (0)

int main() {
    const size_t bytes = 1 << 28;                 // 256 MB
    const int iters = 5;

    float *d_data;
    CHECK(cudaMalloc((void **)&d_data, bytes));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ---- 方式一:普通可换页内存 (malloc) ----
    float *h_pageable = (float *)malloc(bytes);
    // 预热一次,排除首次传输的初始化开销
    CHECK(cudaMemcpy(d_data, h_pageable, bytes, cudaMemcpyHostToDevice));
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i)
        CHECK(cudaMemcpy(d_data, h_pageable, bytes, cudaMemcpyHostToDevice));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float pageable_ms;
    cudaEventElapsedTime(&pageable_ms, start, stop);
    printf("malloc (pageable) H2D: %.2f ms/iter  (%.2f GB/s)\n",
           pageable_ms / iters, iters * bytes / 1e9 / (pageable_ms / 1000));

    // ---- 方式二:页锁定内存 (cudaMallocHost) ----
    float *h_pinned;
    CHECK(cudaMallocHost((void **)&h_pinned, bytes));
    CHECK(cudaMemcpy(d_data, h_pinned, bytes, cudaMemcpyHostToDevice));
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i)
        CHECK(cudaMemcpy(d_data, h_pinned, bytes, cudaMemcpyHostToDevice));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float pinned_ms;
    cudaEventElapsedTime(&pinned_ms, start, stop);
    printf("pinned H2D           : %.2f ms/iter  (%.2f GB/s)  speedup %.1fx\n",
           pinned_ms / iters, iters * bytes / 1e9 / (pinned_ms / 1000),
           pageable_ms / pinned_ms);

    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i)
        CHECK(cudaMemcpy(h_pageable, d_data, bytes, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    pageable_ms = 0;
    cudaEventElapsedTime(&pageable_ms, start, stop);
    printf("malloc (pageable) D2H: %.2f ms/iter  (%.2f GB/s)\n",
            pageable_ms / iters, iters * bytes / 1e9 / (pageable_ms / 1000));

    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i)
        CHECK(cudaMemcpy(h_pinned, d_data, bytes, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    pinned_ms = 0;
    cudaEventElapsedTime(&pinned_ms, start, stop);
    printf("pinned D2H           : %.2f ms/iter  (%.2f GB/s)  speedup %.1fx\n",
            pinned_ms / iters, iters * bytes / 1e9 / (pinned_ms / 1000),
            pageable_ms / pinned_ms);

    CHECK(cudaFreeHost(h_pinned));
    free(h_pageable);
    cudaFree(d_data);
    return 0;
}

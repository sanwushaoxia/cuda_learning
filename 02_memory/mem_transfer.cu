// mem_transfer.cu — 显存管理生命周期与传输耗时测量
//
// 学习点:
//   1. cudaMalloc / cudaMemcpy / cudaFree 完整生命周期
//   2. H2D 与 D2H 传输速度差异(D2H 通常更慢,涉及写回与中断)
//   3. 用 cudaEvent 精确给 GPU 侧操作计时
//   4. 传输带宽估算:RTX 3060 走 PCIe 4.0 x4 理论约 8 GB/s
//
// 编译运行: make run T=mem_transfer
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); exit(1); } \
} while (0)

int main() {
    const size_t bytes = 1 << 28;                 // 256 MB
    float *h_data = (float *)malloc(bytes);
    for (size_t i = 0; i < bytes / sizeof(float); ++i) h_data[i] = 1.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ---- 显存分配与释放 ----
    float *d_data;
    CHECK(cudaMalloc((void **)&d_data, bytes));
    printf("已分配 %.0f MB 显存\n", bytes / 1048576.0);

    // ---- H2D 传输计时 ----
    cudaEventRecord(start);
    CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float h2d_ms;
    cudaEventElapsedTime(&h2d_ms, start, stop);
    printf("H2D: %.2f ms  (有效带宽 %.2f GB/s)\n",
           h2d_ms, bytes / 1e9 / (h2d_ms / 1000));

    // ---- D2H 传输计时 ----
    cudaEventRecord(start);
    CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float d2h_ms;
    cudaEventElapsedTime(&d2h_ms, start, stop);
    printf("D2H: %.2f ms  (有效带宽 %.2f GB/s)\n",
           d2h_ms, bytes / 1e9 / (d2h_ms / 1000));

    // ---- kernel 计时对照:拷贝 256MB 的耗时 vs 计算它多快 ----
    cudaEventRecord(start);
    CHECK(cudaMemset(d_data, 0, bytes));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float memset_ms;
    cudaEventElapsedTime(&memset_ms, start, stop);
    printf("设备内 memset 同样 256MB 只需 %.2f ms —— 显存带宽远高于 PCIe\n", memset_ms);

    cudaFree(d_data);
    free(h_data);
    return 0;
}

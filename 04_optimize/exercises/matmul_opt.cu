// matmul_opt.cu — 练习:把阶段三的 tiled matmul 优化到位
//
// 在阶段三 tiled 基础上逐步优化,每步都要用 make profile T=matmul_opt
// 或 ncu 观察指标变化。目标:在 1024x1024 矩阵上明显超过 tiled 基线。
//
// 优化清单(按难度递增):
//   1. 每 block 计算 64x64 输出,每线程算 2x2(减少全局内存流量)
//   2. 寄存器分块:每线程用寄存器累加多个结果(thread coarsening)
//   3. A tile 一维载入(sA 只需 [TILE][TILE+1] 的一半线程搬运)
//   4. 对比 ncu 指标:sm__throughput、dram__bytes、occupancy
//
// 提示:分步做,每步校验正确性!
//
// 完成后运行: make run T=matmul_opt
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); exit(1); } \
} while (0)

#define N 1024

// TODO: 从阶段三复制 tiled matmul 作为起点,逐步应用优化清单
__global__ void matmul_opt(const float *A, const float *B, float *C, int n) {
    // 在此写代码
}

int main() {
    const size_t bytes = sizeof(float) * N * N;
    float *h_A = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);
    for (int i = 0; i < N * N; ++i) h_A[i] = 1.0f;

    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((void **)&d_A, bytes));
    CHECK(cudaMalloc((void **)&d_B, bytes));
    CHECK(cudaMalloc((void **)&d_C, bytes));
    CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_A, bytes, cudaMemcpyHostToDevice));

    // TODO: 启动 kernel,计时(多次取平均),校验 C[i] == N
    printf("matmul_opt: 尚未实现\n");
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_C);
    return 0;
}

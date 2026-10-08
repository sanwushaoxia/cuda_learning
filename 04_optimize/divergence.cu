// divergence.cu — warp 分支发散 (warp divergence):SIMT 的代价
//
// 学习点:
//   1. warp 内 32 线程锁步执行,遇到 if 分支时"都执行,各自掩码" ——
//      两个分支都执行,总耗时 = 最长分支
//   2. 发散模式:相邻线程走不同分支(最坏) vs 整 warp 走同一分支(无代价)
//   3. 本例对比:交替分支 vs 按 warp 对齐分支
//
// 编译运行: make run T=divergence
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); exit(1); } \
} while (0)

#define N (1 << 26)

// 模拟贵分支:大量浮点迭代
__device__ float expensive(float x) {
    for (int k = 0; k < 32; ++k) x = fmaf(x, 1.0001f, 0.001f);
    return x;
}

// 相邻线程交替走分支:warp 内每次迭代都发散
__global__ void divergent(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (i % 2 == 0)
            out[i] = expensive(in[i]);
        else
            out[i] = in[i] + 1.0f;
    }
}

// 按 warp 对齐分支:整个 warp(32 线程)走同一分支,无发散
__global__ void warp_aligned(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if ((i / 32) % 2 == 0)
            out[i] = expensive(in[i]);
        else
            out[i] = in[i] + 1.0f;
    }
}

float bench(void (*kernel)(const float *, float *, int),
            const float *d_in, float *d_out) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    kernel<<<N / 256, 256>>>(d_in, d_out, N);
    CHECK(cudaDeviceSynchronize());
    float total = 0;
    for (int it = 0; it < 10; ++it) {
        cudaEventRecord(start);
        kernel<<<N / 256, 256>>>(d_in, d_out, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        total += ms;
    }
    return total / 10;
}

int main() {
    const size_t bytes = sizeof(float) * N;
    float *d_in, *d_out;
    CHECK(cudaMalloc((void **)&d_in, bytes));
    CHECK(cudaMalloc((void **)&d_out, bytes));
    CHECK(cudaMemset(d_in, 1, bytes));

    printf("  发散分支(线程交替): %.3f ms\n", bench(divergent, d_in, d_out));
    printf("  对齐分支(warp 整齐): %.3f ms\n", bench(warp_aligned, d_in, d_out));

    cudaFree(d_in); cudaFree(d_out);
    return 0;
}

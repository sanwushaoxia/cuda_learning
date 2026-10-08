// coalesce.cu — 合并访存 (coalescing):决定带宽利用率的第一因素
//
// 学习点:
//   1. 同一 warp 的 32 线程访问连续地址 -> 硬件合并成少量事务,满带宽
//   2. 访问跨步地址 -> 一个事务只能服务少数线程,带宽利用率暴跌
//   3. 本例用同一份数据的两种读法演示:连续 vs 步长 32
//   4. 判断标准:nvprof/ncu 里的 gld_efficiency / 内存事务数
//
// 编译运行: make run T=coalesce
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); exit(1); } \
} while (0)

#define N (1 << 26)

// 合并访存:warp 内线程 i 读 in[blockDim*blockIdx + tid],连续地址
__global__ void read_coalesced(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * 2.0f;
}

// 非合并访存:步长 32 跳跃读取(每个 128B 事务只用 1 个 float)
__global__ void read_strided(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int j = (i * 32) % n;     // 保证不越界,但地址不再连续
        out[i] = in[j] * 2.0f;
    }
}

float bench(void (*kernel)(const float *, float *, int),
            const float *d_in, float *d_out, int iters = 20) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    kernel<<<N / 256, 256>>>(d_in, d_out, N);   // 预热
    CHECK(cudaDeviceSynchronize());

    float total = 0;
    for (int it = 0; it < iters; ++it) {
        cudaEventRecord(start);
        kernel<<<N / 256, 256>>>(d_in, d_out, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        total += ms;
    }
    return total / iters;
}

int main() {
    const size_t bytes = sizeof(float) * N;
    float *d_in, *d_out;
    CHECK(cudaMalloc((void **)&d_in, bytes));
    CHECK(cudaMalloc((void **)&d_out, bytes));
    CHECK(cudaMemset(d_in, 1, bytes));

    float ms_c = bench(read_coalesced, d_in, d_out);
    float ms_s = bench(read_strided, d_in, d_out);

    // 读 + 写 = 2 * bytes 的流量
    printf("N=%d, 每次 kernel 读+写 %.0f MB\n", N, 2 * bytes / 1048576.0);
    printf("  连续访问: %.3f ms  (有效带宽 %5.0f GB/s)\n",
           ms_c, 2 * bytes / 1e9 / (ms_c / 1000));
    printf("  跨步访问: %.3f ms  (有效带宽 %5.0f GB/s)\n",
           ms_s, 2 * bytes / 1e9 / (ms_s / 1000));
    printf("  慢了 %.1f 倍 —— 算法没变,只是地址模式变了\n", ms_s / ms_c);

    cudaFree(d_in); cudaFree(d_out);
    return 0;
}

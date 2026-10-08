// multi_stream.cu — 多流并发执行独立 kernel + event 计时与依赖
//
// 学习点:
//   1. 多个独立 kernel 放入不同流,SM 空闲资源被第二个 kernel 填充
//   2. cudaEvent:跨流依赖(等待/记录),比 cudaStreamSynchronize 更细粒度
//   3. default stream 的隐式同步陷阱:legacy 默认流与所有流互斥
//
// 编译运行: make run T=multi_stream
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); exit(1); } \
} while (0)

#define N (1 << 24)

// 计算量较大的 kernel,便于观察并发重叠
__global__ void busy_kernel(float *data, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = data[i];
        for (int k = 0; k < iters; ++k) v = fmaf(v, 1.0000001f, 1e-7f);
        data[i] = v;
    }
}

int main() {
    const size_t bytes = sizeof(float) * N;
    float *d_a, *d_b;
    CHECK(cudaMalloc((void **)&d_a, bytes));
    CHECK(cudaMalloc((void **)&d_b, bytes));
    CHECK(cudaMemset(d_a, 1, bytes));
    CHECK(cudaMemset(d_b, 1, bytes));

    cudaStream_t s1, s2;
    CHECK(cudaStreamCreate(&s1));
    CHECK(cudaStreamCreate(&s2));
    cudaEvent_t start, stop, e1;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventCreate(&e1));

    const dim3 grid(N / 256), block(256);

    // ---- 串行:两个 kernel 同流 ----
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(start);
    busy_kernel<<<grid, block, 0, s1>>>(d_a, N, 1000);
    busy_kernel<<<grid, block, 0, s1>>>(d_b, N, 1000);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float serial_ms;
    cudaEventElapsedTime(&serial_ms, start, stop);

    // ---- 并发:两个 kernel 不同流 ----
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(start);
    busy_kernel<<<grid, block, 0, s1>>>(d_a, N, 1000);
    busy_kernel<<<grid, block, 0, s2>>>(d_b, N, 1000);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float parallel_ms;
    cudaEventElapsedTime(&parallel_ms, start, stop);

    // ---- 跨流依赖:s2 的 kernel 等 s1 的 kernel 完成 ----
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(start);
    busy_kernel<<<grid, block, 0, s1>>>(d_a, N, 1000);
    CHECK(cudaEventRecord(e1, s1));                 // 在 s1 里打标记
    CHECK(cudaStreamWaitEvent(s2, e1));             // s2 等待该标记
    busy_kernel<<<grid, block, 0, s2>>>(d_b, N, 1000);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float dep_ms;
    cudaEventElapsedTime(&dep_ms, start, stop);

    printf("单 kernel 约 %.2f ms\n", serial_ms / 2);
    printf("  同流串行     : %7.2f ms (两倍)\n", serial_ms);
    printf("  双流并发     : %7.2f ms (单 kernel 未能占满 SM 时更快)\n", parallel_ms);
    printf("  跨流依赖执行 : %7.2f ms (WaitEvent 保证顺序)\n", dep_ms);

    CHECK(cudaStreamDestroy(s1));
    CHECK(cudaStreamDestroy(s2));
    cudaFree(d_a); cudaFree(d_b);
    return 0;
}

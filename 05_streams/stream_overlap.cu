// stream_overlap.cu — 计算与传输重叠:流的核心应用
//
// 学习点:
//   1. 单流世界:H2D -> kernel -> D2H 串行执行,PCIe 与 SM 总有一个闲置
//   2. 分块 + 双流:chunk i 计算时,chunk i+1 正在传输 -> 流水线
//   3. 前提:pinned host 内存 + 非 default 流(默认流会与其它操作互斥)
//
// 编译运行: make run T=stream_overlap
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); exit(1); } \
} while (0)

__global__ void scale_kernel(float *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = data[i] * 2.0f + 1.0f;
}

float bench(int n_streams, int n, float *h_data, float *d_data) {
    const int chunk = n / n_streams;              // 每 stream 处理的分块
    const size_t chunk_bytes = sizeof(float) * chunk;
    const int block = 256;

    cudaStream_t streams[8];
    for (int s = 0; s < n_streams; ++s) CHECK(cudaStreamCreate(&streams[s]));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(start);

    for (int s = 0; s < n_streams; ++s) {
        float *h_chunk = h_data + s * chunk;
        float *d_chunk = d_data + s * chunk;
        // 三个操作进同一 stream:stream 内保序,stream 间并行
        CHECK(cudaMemcpyAsync(d_chunk, h_chunk, chunk_bytes,
                              cudaMemcpyHostToDevice, streams[s]));
        scale_kernel<<<(chunk + block - 1) / block, block, 0, streams[s]>>>(d_chunk, chunk);
        CHECK(cudaMemcpyAsync(h_chunk, d_chunk, chunk_bytes,
                              cudaMemcpyDeviceToHost, streams[s]));
    }

    cudaEventRecord(stop);
    CHECK(cudaEventSynchronize(stop));
    float ms; cudaEventElapsedTime(&ms, start, stop);

    for (int s = 0; s < n_streams; ++s) cudaStreamDestroy(streams[s]);
    return ms;
}

int main() {
    const int N = 1 << 26;
    const size_t bytes = sizeof(float) * N;

    // 必须用 pinned 内存,否则 async 传输会退化成同步
    float *h_data;
    CHECK(cudaMallocHost((void **)&h_data, bytes));
    for (int i = 0; i < N; ++i) h_data[i] = 1.0f;

    float *d_data;
    CHECK(cudaMalloc((void **)&d_data, bytes));

    // 正确性基线:单流跑完,数据应全为 3.0f(2*1+1)
    float ms1 = bench(1, N, h_data, d_data);
    int bad = 0;
    for (int i = 0; i < N; ++i) if (h_data[i] != 3.0f) bad++;

    float ms2 = bench(2, N, h_data, d_data);
    float ms4 = bench(4, N, h_data, d_data);

    printf("N=%d, pinned 内存\n", N);
    printf("  1 流(串行)   : %7.2f ms  (校验 %s)\n", ms1, bad == 0 ? "✓" : "✗");
    printf("  2 流(流水线) : %7.2f ms\n", ms2);
    printf("  4 流(流水线) : %7.2f ms\n", ms4);
    printf("  流水线让传输与计算重叠,接近两者较大值而非三者相加\n");

    CHECK(cudaFree(d_data));
    CHECK(cudaFreeHost(h_data));
    return 0;
}

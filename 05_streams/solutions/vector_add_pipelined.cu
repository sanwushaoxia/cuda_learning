// vector_add_pipelined.cu — 练习参考解答:分块 + 双流流水线化 vector_add
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); exit(1); } \
} while (0)

#define N (1 << 24)

__global__ void add_kernel(const float *x, const float *y, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] + y[i];
}

// n_streams: 1 = 串行基线,2 = 流水线
float bench(int n_streams, const float *h_x, const float *h_y, float *h_out,
            float *d_x, float *d_y, float *d_out) {
    const int chunk = N / n_streams;
    const size_t chunk_bytes = sizeof(float) * chunk;
    const int block = 256;

    cudaStream_t streams[2];
    for (int s = 0; s < n_streams; ++s) CHECK(cudaStreamCreate(&streams[s]));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(start);

    for (int s = 0; s < n_streams; ++s) {
        const float *hx = h_x + s * chunk;
        const float *hy = h_y + s * chunk;
        float *ho = h_out + s * chunk;
        float *dx = d_x + s * chunk;
        float *dy = d_y + s * chunk;
        float *doo = d_out + s * chunk;

        CHECK(cudaMemcpyAsync(dx, hx, chunk_bytes, cudaMemcpyHostToDevice, streams[s]));
        CHECK(cudaMemcpyAsync(dy, hy, chunk_bytes, cudaMemcpyHostToDevice, streams[s]));
        add_kernel<<<(chunk + block - 1) / block, block, 0, streams[s]>>>(dx, dy, doo, chunk);
        CHECK(cudaMemcpyAsync(ho, doo, chunk_bytes, cudaMemcpyDeviceToHost, streams[s]));
    }

    cudaEventRecord(stop);
    CHECK(cudaEventSynchronize(stop));
    float ms; cudaEventElapsedTime(&ms, start, stop);

    for (int s = 0; s < n_streams; ++s) cudaStreamDestroy(streams[s]);
    return ms;
}

int main() {
    const size_t bytes = sizeof(float) * N;

    float *h_x, *h_y, *h_out;
    CHECK(cudaMallocHost((void **)&h_x, bytes));
    CHECK(cudaMallocHost((void **)&h_y, bytes));
    CHECK(cudaMallocHost((void **)&h_out, bytes));
    for (int i = 0; i < N; ++i) {
        h_x[i] = 1.0f;
        h_y[i] = 2.0f;
    }

    float *d_x, *d_y, *d_out;
    CHECK(cudaMalloc((void **)&d_x, bytes));
    CHECK(cudaMalloc((void **)&d_y, bytes));
    CHECK(cudaMalloc((void **)&d_out, bytes));

    float ms1 = bench(1, h_x, h_y, h_out, d_x, d_y, d_out);
    int bad1 = 0;
    for (int i = 0; i < N; ++i) if (h_out[i] != 3.0f) bad1++;

    float ms2 = bench(2, h_x, h_y, h_out, d_x, d_y, d_out);
    int bad2 = 0;
    for (int i = 0; i < N; ++i) if (h_out[i] != 3.0f) bad2++;

    printf("N=%d\n", N);
    printf("  1 stream : %7.2f ms  (check %s)\n", ms1, bad1 == 0 ? "OK" : "FAIL");
    printf("  2-stream pipe : %7.2f ms  (check %s)\n", ms2, bad2 == 0 ? "OK" : "FAIL");

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_out);
    cudaFreeHost(h_x); cudaFreeHost(h_y); cudaFreeHost(h_out);
    return 0;
}

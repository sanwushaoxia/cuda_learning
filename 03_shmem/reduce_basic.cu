// reduce_basic.cu — 并行归约(求和)的两种经典实现
//
// 学习点:
//   1. 归约是"线程协作"入门首选:1M 个数求和,256 线程如何合作?
//   2. 相邻配对 (interleaved pairing):步长折半,stride 相邻元素相加,
//      直观但有大量空闲线程
//   3. 交错配对 (sequential addressing):每线程只和固定距离元素相加,
//      无空闲线程,活跃线程集中排布,是共享内存归约的标准写法
//
// 编译运行: make run T=reduce_basic
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); exit(1); } \
} while (0)

#define BLOCK_SIZE 256

// ---- 相邻配对:偶数下标与右邻相加,步长每轮减半 ----
__global__ void reduce_interleaved(const float *in, float *out, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        // 问题:stride 越大,越多线程没事做(warp 内分支浪费)
        if (tid % (2 * stride) == 0)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// ---- 交错配对:前半元素与后半元素配对,无空闲线程 ----
__global__ void reduce_sequential(const float *in, float *out, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// CPU 上递归折半求和 block 部分和,得到最终结果
float finish_reduction(const float *partial, int count) {
    float *tmp = (float *)malloc(sizeof(float) * count);
    memcpy(tmp, partial, sizeof(float) * count);
    while (count > 1) {
        int half = count / 2;
        for (int i = 0; i < half; ++i) tmp[i] += tmp[i + half];
        if (count % 2) tmp[0] += tmp[count - 1];
        count = half;
    }
    float result = tmp[0];
    free(tmp);
    return result;
}

float run_kernel(void (*kernel)(const float *, float *, int),
                 const float *d_in, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float *d_partial, *h_partial;
    cudaMalloc((void **)&d_partial, sizeof(float) * blocks);
    kernel<<<blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_in, d_partial, n);
    CHECK(cudaDeviceSynchronize());
    h_partial = (float *)malloc(sizeof(float) * blocks);
    cudaMemcpy(h_partial, d_partial, sizeof(float) * blocks, cudaMemcpyDeviceToHost);
    float sum = finish_reduction(h_partial, blocks);
    cudaFree(d_partial); free(h_partial);
    return sum;
}

int main() {
    const int N = 1 << 24;
    const size_t bytes = sizeof(float) * N;
    float *h_in = (float *)malloc(bytes);
    for (int i = 0; i < N; ++i) h_in[i] = 1.0f;      // 全 1,和应为 N

    float *d_in;
    cudaMalloc((void **)&d_in, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    float s1 = run_kernel(reduce_interleaved, d_in, N);
    float s2 = run_kernel(reduce_sequential, d_in, N);
    printf("N=%d (期望和 %d)\n", N, N);
    printf("  相邻配对: %.0f  %s\n", s1, s1 == (float)N ? "✓" : "✗");
    printf("  交错配对: %.0f  %s\n", s2, s2 == (float)N ? "✓" : "✗");

    cudaFree(d_in); free(h_in);
    return 0;
}

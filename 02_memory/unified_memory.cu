// unified_memory.cu — 统一内存 (cudaMallocManaged)
//
// 学习点:
//   1. 统一内存:一块 CPU/GPU 都能直接访问的指针,页按需自动迁移
//   2. 代码大幅简化:不再手写 cudaMemcpy
//   3. 代价:首次访问触发 page fault 迁移,访问模式复杂时性能不可控
//   4. cudaMemPrefetchAsync 可提前把数据推到 GPU 减少缺页
//
// 编译运行: make run T=unified_memory
#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); exit(1); } \
} while (0)

__global__ void scale_kernel(float *data, float factor, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= factor;
}

int main() {
    const int N = 1 << 24;
    const size_t bytes = sizeof(float) * N;

    // 一条调用,同时拿到 CPU/GPU 可用的指针
    float *data;
    CHECK(cudaMallocManaged((void **)&data, bytes));

    // CPU 侧初始化 —— 此时数据页在 host
    for (int i = 0; i < N; ++i) data[i] = 1.0f;

    // 预取到 GPU:提前批量迁移,避免 kernel 里逐页缺页
    int device;
    cudaGetDevice(&device);
    CHECK(cudaMemPrefetchAsync(data, bytes, device));

    const int block_size = 256;
    const int grid_size = (N + block_size - 1) / block_size;
    scale_kernel<<<grid_size, block_size>>>(data, 2.0f, N);
    CHECK(cudaDeviceSynchronize());

    // 访问前把数据预取回 CPU 侧,再校验
    CHECK(cudaMemPrefetchAsync(data, bytes, cudaCpuDeviceId));
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        if (data[i] != 2.0f) errors++;
    }
    printf("unified memory N=%d: %s\n", N, errors == 0 ? "all elements OK" : "FAILED");

    CHECK(cudaFree(data));
    return 0;
}

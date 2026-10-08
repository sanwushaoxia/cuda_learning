// occupancy.cu — 占用率 (occupancy) 与 block 尺寸扫描
//
// 学习点:
//   1. 占用率 = 活跃 warp 数 / SM 最大 warp 数,过低时延迟无法被并行度掩盖
//   2. block 尺寸影响:寄存器/共享内存占用、warp 数、同步开销
//   3. cudaOccupancyMaxActiveBlocksPerMultiprocessor 查询理论占用
//   4. 经验:不要迷信占用率 100%,扫一遍找实测最优点才是正道
//
// 编译运行: make run T=occupancy
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); exit(1); } \
} while (0)

#define N (1 << 26)

// 中等计算强度的 kernel:3 次乘加 + grid-stride
__global__ void workload(const float *in, float *out, int n) {
    float acc = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * blockDim.x) {
        float v = in[i];
        v = v * 1.1f + 0.2f;
        v = v * 1.2f - 0.3f;
        acc += v * 1.3f;
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

int main() {
    const size_t bytes = sizeof(float) * N;
    float *d_in, *d_out;
    CHECK(cudaMalloc((void **)&d_in, bytes));
    CHECK(cudaMalloc((void **)&d_out, bytes));
    CHECK(cudaMemset(d_in, 1, bytes));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    printf("%s: 每 SM 最多 %d 线程 (%d warps), 共 %d SM\n\n",
           prop.name, prop.maxThreadsPerMultiProcessor,
           prop.maxThreadsPerMultiProcessor / 32, prop.multiProcessorCount);

    printf("%-10s %-12s %-10s %-12s\n", "block", "grid", "耗时ms", "有效GB/s");
    for (int block_size = 32; block_size <= 1024; block_size *= 2) {
        // 固定总线程规模,只变 block 形状
        const int total_threads = N / 4;
        int grid_size = total_threads / block_size;

        // 理论占用查询
        int max_active_blocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_active_blocks, workload, block_size, 0);
        float occupancy = (float)max_active_blocks * block_size /
                          prop.maxThreadsPerMultiProcessor;

        // 计时
        workload<<<grid_size, block_size>>>(d_in, d_out, N);   // 预热
        CHECK(cudaDeviceSynchronize());
        cudaEventRecord(start);
        workload<<<grid_size, block_size>>>(d_in, d_out, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);

        printf("%-10d %-12d %-10.3f %-12.0f (理论占用 %.0f%%)\n",
               block_size, grid_size, ms, 2 * bytes / 1e9 / (ms / 1000),
               occupancy * 100);
    }

    cudaFree(d_in); cudaFree(d_out);
    return 0;
}

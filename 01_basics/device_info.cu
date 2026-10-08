// device_info.cu — 查询 GPU 设备属性,认识运行环境
// 学习点: cudaGetDeviceCount / cudaGetDeviceProperties / CUDA_CHECK 错误检查宏
//
// 编译运行: make run T=device_info
#include <stdio.h>
#include "cuda_common.cuh"

int main() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    printf("检测到 %d 个 CUDA 设备\n\n", device_count);

    for (int dev = 0; dev < device_count; ++dev) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        printf("=== 设备 %d: %s ===\n", dev, prop.name);
        printf("  计算能力          : %d.%d\n", prop.major, prop.minor);
        printf("  SM 数量           : %d\n", prop.multiProcessorCount);
        printf("  每 SM 最大线程数  : %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  每 block 最大线程 : %d\n", prop.maxThreadsPerBlock);
        printf("  block 最大维度    : (%d, %d, %d)\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  grid 最大维度     : (%d, %d, %d)\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  全局内存          : %.1f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  每 block 共享内存 : %.1f KB\n", prop.sharedMemPerBlock / 1024.0);
        printf("  L2 缓存           : %.1f MB\n", prop.l2CacheSize / (1024.0 * 1024.0));
        printf("  时钟频率          : %.2f GHz\n", prop.clockRate / 1e6);
        printf("  warp 大小         : %d\n", prop.warpSize);
        printf("\n");
    }
    return 0;
}

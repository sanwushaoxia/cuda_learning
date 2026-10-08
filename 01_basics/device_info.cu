// device_info.cu — 查询 GPU 设备属性,认识运行环境
// 学习点: cudaGetDeviceCount / cudaGetDeviceProperties / CUDA_CHECK 错误检查宏
//
// 编译运行: make run T=device_info
#include <stdio.h>
#include "cuda_common.cuh"

int main() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    printf("Found %d CUDA device(s)\n\n", device_count);

    for (int dev = 0; dev < device_count; ++dev) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        // name: 设备名(营销型号)
        printf("=== Device %d: %s ===\n", dev, prop.name);
        // major.minor: 计算能力(架构世代),决定指令集与硬件特性
        printf("  Compute capability : %d.%d\n", prop.major, prop.minor);
        // multiProcessorCount: SM 数量,并行度的硬件上限之一
        printf("  SM count           : %d\n", prop.multiProcessorCount);
        // maxThreadsPerMultiProcessor: 单 SM 同时常驻的最大线程数(占用率相关)
        printf("  Max threads/SM     : %d\n", prop.maxThreadsPerMultiProcessor);
        // maxThreadsPerBlock: 单 block 允许的最大线程数(通常 1024)
        printf("  Max threads/block  : %d\n", prop.maxThreadsPerBlock);
        // maxThreadsDim: block 在 x/y/z 各维的最大线程数
        printf("  Max block dims     : (%d, %d, %d)\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        // maxGridSize: grid 在 x/y/z 各维的最大 block 数
        printf("  Max grid dims      : (%d, %d, %d)\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        // totalGlobalMem: 全局显存容量
        printf("  Global memory      : %.1f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        // sharedMemPerBlock: 每 block 可用共享内存上限(静态+动态合计)
        printf("  Shared mem/block   : %.1f KB\n", prop.sharedMemPerBlock / 1024.0);
        // l2CacheSize: 片上 L2 缓存大小,缓解全局内存延迟
        printf("  L2 cache           : %.1f MB\n", prop.l2CacheSize / (1024.0 * 1024.0));
        // clockRate: SM 时钟频率(kHz),越高峰值算力通常越高
        printf("  Clock rate         : %.2f GHz\n", prop.clockRate / 1e6);
        // warpSize: 调度与执行的基本单位,几乎总是 32
        printf("  Warp size          : %d\n", prop.warpSize);
        printf("\n");
    }
    return 0;
}

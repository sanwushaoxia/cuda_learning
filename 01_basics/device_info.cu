// device_info.cu — 查询 GPU 设备属性,认识运行环境
// 学习点: cudaGetDeviceCount / cudaGetDeviceProperties /
//         cudaDeviceGetAttribute / CUDA_CHECK 错误检查宏
//
// README「硬件执行模型」「共享内存约束」等小节引用的数字(48 warp 槽位、
// 16 blocks/SM、共享内存 28+100 分界、持久 L2 窗口等)全部可在此对照。
//
// 字段分两类:
//   1. prop.xxx — cudaDeviceProp 结构体字段,cudaGetDeviceProperties 一次取回;
//   2. cudaDeviceGetAttribute — 结构体没有的字段(如 Max blocks/SM)逐个查。
//
// 编译运行: make run T=device_info
#include <stdio.h>
#include "../cuda_common.cuh"

// 查询一个结构体里没有的属性并返回(cudaDeviceGetAttribute 封装)
static int query_attr(cudaDeviceAttr attr, int dev) {
    int v = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&v, attr, dev));
    return v;
}

int main() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    printf("Found %d CUDA device(s)\n\n", device_count);

    for (int dev = 0; dev < device_count; ++dev) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        // ---- 身份与算力 --------------------------------------------------
        // name: 设备名(营销型号)
        printf("=== Device %d: %s ===\n", dev, prop.name);
        // major.minor: 计算能力(架构世代),决定指令集与硬件特性
        printf("  Compute capability : %d.%d\n", prop.major, prop.minor);
        // multiProcessorCount: SM 数量,并行度的硬件上限之一
        printf("  SM count           : %d\n", prop.multiProcessorCount);

        // ---- 执行单元与调度(README「SM 内部」一节的数字) ---------------
        // warpSize: 调度与执行的基本单位,几乎总是 32
        printf("  Warp size          : %d\n", prop.warpSize);
        // maxThreadsPerMultiProcessor: 单 SM 同时驻留的最大线程数
        // 除以 warpSize = warp 槽位数(1536/32 = 48,4 调度器各管 12)
        printf("  Max threads/SM     : %d      ← %d 个 warp 槽位\n",
               prop.maxThreadsPerMultiProcessor,
               prop.maxThreadsPerMultiProcessor / prop.warpSize);
        // maxThreadsPerBlock: 单 block 允许的最大线程数(通常 1024)
        printf("  Max threads/block  : %d\n", prop.maxThreadsPerBlock);
        // maxThreadsDim: block 在 x/y/z 各维的最大线程数(512/512/64)
        printf("  Max block dims     : (%d, %d, %d)\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        // maxGridSize: grid 在 x/y/z 各维的最大 block 数(2^31-1/65535/65535)
        printf("  Max grid dims      : (%d, %d, %d)\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        // maxBlocksPerMultiProcessor: 单 SM 同时驻留的 block 数硬上限
        // (GA10x 为 16)。结构体没有这个字段,要用 attribute 单独查。
        // 与 warp 槽位共同决定占用率: block 小被它卡住(16×1 warp,
        // 48 槽空 2/3); block 大被槽位卡住(1024 线程占 32 槽,驻留 1 个)。
        printf("  Max blocks/SM      : %d\n",
               query_attr(cudaDevAttrMaxBlocksPerMultiprocessor, dev));
        // regsPerBlock: 单 block 可用的 32-bit 寄存器总数(与 64K/SM 对应)
        printf("  Regs/block         : %d\n", prop.regsPerBlock);
        // regsPerMultiprocessor: 单 SM 寄存器堆总量(64K × 32-bit)
        printf("  Regs/SM            : %d      ← 寄存器堆总量\n",
               query_attr(cudaDevAttrMaxRegistersPerMultiprocessor, dev));

        // ---- 存储层次(README「共享内存约束」「L2 vs 显存」的数字) -------
        // totalGlobalMem: 全局显存容量(数据的"老家")
        printf("  Global memory      : %.1f GB\n",
               prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        // memoryBusWidth + memoryClockRate: 理论峰值带宽 =
        //   busWidth(字节)/8 × clock(×2,DDR 双沿) × 2 → 约 360 GB/s
        printf("  Mem bus width      : %d-bit\n", prop.memoryBusWidth);
        printf("  Mem clock rate     : %.2f GHz\n", prop.memoryClockRate / 1e6);
        // l2CacheSize: 片上 L2 缓存(全体 SM 共享一个,跨 SM 通信枢纽)
        printf("  L2 cache           : %.1f MB\n", prop.l2CacheSize / (1024.0 * 1024.0));
        // persistingL2CacheMaxSize: 可划给持久 L2 窗口的上限(30 系起支持)
        printf("  Persisting L2 max  : %.1f MB\n",
               query_attr(cudaDevAttrMaxPersistingL2CacheSize, dev) / (1024.0 * 1024.0));
        // l2CacheSize 里能分给共享内存的部分之外的划分:CC 8.6 上 L1 与
        // 共享内存的分界固定(L1 28 KB + smem 100 KB = 128 KB),不可配置
        printf("  L1/shared total    : %.1f KB\n",
               query_attr(cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev) / 1024.0 + 28.0);
        // sharedMemPerMultiprocessor: 每 SM 上所有驻留 block 的共享内存之和上限
        printf("  Shared mem/SM      : %.1f KB\n",
               query_attr(cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev) / 1024.0);
        // sharedMemPerBlock: 单 block 默认共享内存上限(不 opt-in 能申请的量)
        printf("  Shared mem/block   : %.1f KB\n", prop.sharedMemPerBlock / 1024.0);
        // sharedMemPerBlockOptin: opt-in 后的单 block 上限
        // (99 KB = 100 KB - 每 block 固定 1 KB 保留)
        printf("  Shared mem opt-in  : %.1f KB\n", prop.sharedMemPerBlockOptin / 1024.0);

        // ---- 频率与时钟 ---------------------------------------------------
        // clockRate: SM 时钟频率(kHz),越高峰值算力通常越高
        printf("  SM clock rate      : %.2f GHz\n", prop.clockRate / 1e6);

        printf("\n");
    }
    return 0;
}

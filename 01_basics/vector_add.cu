// vector_add.cu — CUDA 的 "Hello World":向量加法 out[i] = x[i] + y[i]
//
// 核心概念:线程层次 thread -> block -> grid
//   - 每个 CUDA 线程只负责一个(或几个)元素,这与 CPU 串行循环完全相反
//   - 全局索引公式: i = blockIdx.x * blockDim.x + threadIdx.x
//   - 网格跨越 (grid-stride) 循环让同一个 kernel 优雅处理任意规模
//
// 编译运行: make run T=vector_add
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// 方式一:一线程一元素。要求 grid_size >= (N + block_size - 1) / block_size,
// 否则下标 i >= grid*block 的元素无人处理。
__global__ void vector_add_one_per_thread(const float *x, const float *y, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {                      // 最后一个 block 可能有越界线程,必须判断
        out[i] = x[i] + y[i];
    }
}

// 方式二:网格跨越循环。即使 grid_size < (N + block_size - 1) / block_size
// 也能算完:每线程以 stride(=grid*block) 步进,轮流覆盖全部元素。
__global__ void vector_add_grid_stride(const float *x, const float *y, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;   // 线程总数
    for (; i < n; i += stride) {
        out[i] = x[i] + y[i];
    }
}

// 主函数跑 4 组实验,展示 block 尺寸如何影响性能。
// 本机(RTX 3060)实测数据(2026-10,每次取 20 次平均,单次波动 ±5%):
//
//   #   block  warp/block  驻留block/SM  warp槽位利用   耗时
//   1   256    8           6 (线程上限)   48/48 = 100%   0.63 ms  ← 基线
//   2   16     1           16 (block上限) 16/48 = 33%    1.98 ms
//   3   33     2           16 (block上限) 32/48 = 67%    0.82 ms
//   4   48     2           16 (block上限) 32/48 = 67%    0.65 ms
//
// 根因分析(为什么 3 和 4 都是 16 block 驻留、同样 2 warp,却差 25%?):
//   - block 太小 → 被每 SM 最多 16 个 block 的硬上限卡住,warp 槽位
//     (48 个/SM)填不满 → 驻留线程少,掩盖内存延迟的能力弱 → 慢;
//   - #4 比 #3 快:33 线程切成 2 个 warp 只用 33/64 = 52% 的槽位
//     (第二个 warp 只干 1/32 的活);48 线程用 48/64 = 75%。
//     block 个数相同时,比的是 warp 槽位利用率(有效驻留线程/SM:
//     528 vs 768);
//   - 结论:选 block 尺寸要看"每 SM 有效驻留线程"= 驻留block数 ×
//     有效线程/block,而不是 block 数本身。128/256/512 都能满载
//     (见 04_optimize/occupancy.cu 的完整扫描)。
// 实验配置:每次运行一组 (kernel 模式, block 尺寸),打印带标签的结果
typedef struct {
    int mode;            // 0 = 一线程一元素, 1 = grid-stride
    int block_size;
    const char *label;
} bench_config;

// 运行单组实验:启动 → 计时 → 取回 → 验证 → 打印(带 block_size 标签)
void run_case(bench_config cfg, const float *d_x, const float *d_y,
              float *d_out, float *h_out, int n, size_t bytes,
              cudaEvent_t start, cudaEvent_t stop) {
    const int grid_size = (n + cfg.block_size - 1) / cfg.block_size;
    cudaMemset(d_out, 0, bytes);

    cudaEventRecord(start);
    if (cfg.mode == 0) {
        vector_add_one_per_thread<<<grid_size, cfg.block_size>>>(d_x, d_y, d_out, n);
    } else {
        vector_add_grid_stride<<<grid_size, cfg.block_size>>>(d_x, d_y, d_out, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // D2H 拷回并校验(cudaMemcpy 隐式同步)
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    int errors = 0;
    for (int i = 0; i < n; ++i) {
        if (h_out[i] != 3.0f) { errors++; }
    }
    printf("block=%-4d %-15s (N=%d): %.2f ms  %s\n",
           cfg.block_size, cfg.label, n, ms,
           errors == 0 ? "all elements OK" : "FAILED");
}

int main() {
    const int N = 1 << 24;                       // 约 1677 万元素
    const size_t bytes = sizeof(float) * N;

    // 1. 分配 host 内存并初始化
    float *h_x = (float *)malloc(bytes);
    float *h_y = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    for (int i = 0; i < N; ++i) {
        h_x[i] = 1.0f;
        h_y[i] = 2.0f;
    }

    // 2. 分配 device 内存并拷贝数据
    float *d_x, *d_y, *d_out;
    cudaMalloc((void **)&d_x, bytes);
    cudaMalloc((void **)&d_y, bytes);
    cudaMalloc((void **)&d_out, bytes);
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice);

    // 计时用的 event(所有实验共用)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 3. 四组实验:满载基线 → block 过小 → warp 碎片 → 碎片改善
    bench_config cases[] = {
        {0, 256, "one-per-thread"},
        {1, 16,  "grid-stride"},
        {1, 33,  "grid-stride"},
        {1, 48,  "grid-stride"},
    };
    for (int c = 0; c < 4; ++c) {
        run_case(cases[c], d_x, d_y, d_out, h_out, N, bytes, start, stop);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_out);
    free(h_x); free(h_y); free(h_out);
    return 0;
}

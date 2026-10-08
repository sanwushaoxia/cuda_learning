// vector_add_pipelined.cu — 练习:把 vector_add 流水线化(分块 + 双流)
//
// 任务:阶段一的 vector_add 传输与计算完全串行。请用 2 个流把它流水线化:
//   流 A:传上半块 -> 算上半块 -> 传回上半块
//   流 B:传下半块 -> 算下半块 -> 传回下半块
//
// 要求:
//   1. pinned host 内存(cudaMallocHost)
//   2. cudaMemcpyAsync + kernel 指定流
//   3. 用 cudaEvent 计时,与单流版本对比
//   4. 校验结果正确
//
// 提示:参考 stream_overlap.cu 的 bench() 写法,但这里只有 2 个流、
// 要求自己写出"每流的操作序列"。
//
// 完成后运行: make run T=vector_add_pipelined
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

int main() {
    const size_t bytes = sizeof(float) * N;

    // TODO: cudaMallocHost 分配 h_x/h_y/h_out 并初始化
    // TODO: cudaMalloc 三个 device 缓冲
    // TODO: 创建 2 个流;数据分成上下两半
    // TODO: 对每半:异步传 x、y -> 启动 kernel(指定流)-> 异步传回
    // TODO: 整体计时(cudaEvent),并统计流数 1 vs 2 的耗时
    // TODO: 校验 h_out 全为 3.0f

    printf("vector_add_pipelined: 尚未实现\n");
    return 0;
}

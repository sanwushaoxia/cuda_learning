// transpose_naive.cu — 练习:矩阵转置(naive 版)
//
// 转置: out(row, col) = in(col, row)。这是理解"合并访存"的经典案例:
// 同一个矩阵,读写两个方向必有一个方向是非合并的,阶段三会给出优化版。
//
// 要求:
//   1. 用二维 grid/block 启动(如 32x32 线程/block)
//   2. 实现转置 kernel,并用全局索引换算行主序下标
//   3. 校验结果正确
//
// 完成后运行: make run T=transpose_naive
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); exit(1); } \
} while (0)

#define M 2048   // 行数
#define K 2048   // 列数

// TODO: 实现转置 kernel
__global__ void transpose_naive(const float *in, float *out, int rows, int cols) {
    // in 是 rows x cols,out 是 cols x rows
    // 提示: out(row, col) = in(col, row)
    //       行主序下标: in[r * cols + c]
}

int main() {
    const size_t in_bytes = sizeof(float) * M * K;
    const size_t out_bytes = sizeof(float) * K * M;   // 大小相同,仅为示意

    float *h_in = (float *)malloc(in_bytes);
    float *h_out = (float *)malloc(out_bytes);
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < K; ++c)
            h_in[r * K + c] = (float)(r * K + c);

    float *d_in, *d_out;
    CHECK(cudaMalloc((void **)&d_in, in_bytes));
    CHECK(cudaMalloc((void **)&d_out, out_bytes));
    CHECK(cudaMemcpy(d_in, h_in, in_bytes, cudaMemcpyHostToDevice));

    // TODO: 用 dim3(32, 32) 配置二维 grid/block 启动 kernel
    // TODO: 拷回 h_out 并校验 h_out[c * M + r] == h_in[r * K + c]

    printf("transpose_naive: not implemented yet\n");
    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);
    return 0;
}

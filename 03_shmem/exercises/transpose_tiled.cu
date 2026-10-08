// transpose_tiled.cu — 练习:分块矩阵转置,消除 bank conflict
//
// 背景:02 练习的 naive 转置写显存读合并/写非合并(或反之),且共享内存
// 直接 32x32 方阵会发生 bank conflict。
//
// 要求:
//   1. block 载入 32x32 tile 到 __shared__
//   2. 用 padding(+1 列)消除 bank conflict: sdata[32][32+1]
//   3. 从共享内存写出时保证写合并
//   4. 校验正确性,并可以用 make profile T=transpose_tiled 对比带宽
//
// 完成后运行: make run T=transpose_tiled
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); exit(1); } \
} while (0)

#define N 2048
#define TILE 32

// TODO: 实现 tile 转置 kernel
// 提示:
//   __shared__ float tile[TILE][TILE + 1];   // +1 让相邻行的行首错开 bank
//   读: tile[threadIdx.y][threadIdx.x] = in[row * N + col];
//   写: out[col * N + row] = tile[threadIdx.x][threadIdx.y];
//   两次 __syncthreads() 之间完成
__global__ void transpose_tiled(const float *in, float *out, int n) {
    // 在此写代码
}

int main() {
    const size_t bytes = sizeof(float) * N * N;
    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    for (int i = 0; i < N * N; ++i) h_in[i] = (float)i;

    float *d_in, *d_out;
    CHECK(cudaMalloc((void **)&d_in, bytes));
    CHECK(cudaMalloc((void **)&d_out, bytes));
    CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // TODO: dim3(TILE, TILE) 启动,同步,拷回并校验 out[c][r] == in[r][c]

    printf("transpose_tiled: 尚未实现\n");
    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);
    return 0;
}

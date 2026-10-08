// matmul_opt.cu — 练习参考解答:寄存器分块 + 2x2 线程粗化的矩阵乘
//
// 关键优化:
//   1. 每 block 32x32 线程,计算 64x64 输出 -> 每线程 2x2 寄存器累加
//   2. 全局内存流量减半:每个 A/B 元素从共享内存读出后做 2 次乘加
//   3. padding 消 bank conflict
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); exit(1); } \
} while (0)

#define N 1024
#define TILE 32   // block 32x32,覆盖 64x64 输出

__global__ void matmul_opt(const float *A, const float *B, float *C, int n) {
    // 每 block 载入 1 个 A tile(32x32)和 1 个 B tile(32x64),
    // 输出 64x64:每线程用 ty/tx 对应 A 的一行、B 的两列
    __shared__ float sA[TILE][TILE + 1];
    __shared__ float sB[TILE][2 * TILE + 1];

    // C 子矩阵的左上角
    int row_base = blockIdx.y * 2 * TILE;
    int col_base = blockIdx.x * 2 * TILE;

    // 每线程 2x2 寄存器累加器
    float c00 = 0.0f, c01 = 0.0f, c10 = 0.0f, c11 = 0.0f;

    int tx = threadIdx.x, ty = threadIdx.y;

    for (int t = 0; t < n / TILE; ++t) {
        // A tile:64 行 x 32 列,由 32x32 线程各搬 2 个元素
        sA[ty][tx]        = A[(row_base + ty) * n + t * TILE + tx];
        sA[ty + TILE][tx] = A[(row_base + ty + TILE) * n + t * TILE + tx];
        // B tile:32 行 x 64 列,每线程各搬 2 个元素
        sB[ty][tx]        = B[(t * TILE + ty) * n + col_base + tx];
        sB[ty][tx + TILE] = B[(t * TILE + ty) * n + col_base + tx + TILE];
        __syncthreads();

        // 2x2 乘加:C 的 4 个角分别乘 A 上/下半 x B 左/右半
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            float a0 = sA[ty][k], a1 = sA[ty + TILE][k];
            float b0 = sB[k][tx], b1 = sB[k][tx + TILE];
            c00 += a0 * b0;
            c01 += a0 * b1;
            c10 += a1 * b0;
            c11 += a1 * b1;
        }
        __syncthreads();
    }

    int row = row_base + ty;
    int col = col_base + tx;
    C[row * n + col] = c00;
    C[row * n + col + TILE] = c01;
    C[(row + TILE) * n + col] = c10;
    C[(row + TILE) * n + col + TILE] = c11;
}

int main() {
    const size_t bytes = sizeof(float) * N * N;
    float *h_A = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);
    for (int i = 0; i < N * N; ++i) h_A[i] = 1.0f;

    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((void **)&d_A, bytes));
    CHECK(cudaMalloc((void **)&d_B, bytes));
    CHECK(cudaMalloc((void **)&d_C, bytes));
    CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_A, bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid(N / (2 * TILE), N / (2 * TILE));

    matmul_opt<<<grid, block>>>(d_A, d_B, d_C, N);   // 预热
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    float total = 0;
    for (int it = 0; it < 10; ++it) {
        cudaEventRecord(start);
        matmul_opt<<<grid, block>>>(d_A, d_B, d_C, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        total += ms;
    }
    float ms = total / 10;

    CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    int errors = 0;
    for (int i = 0; i < N * N && errors == 0; ++i)
        if (h_C[i] != (float)N) { errors++; break; }

    printf("matmul_opt (2x2 寄存器分块): %.2f ms  %s (有效 %.0f GFLOPS)\n",
           ms, errors == 0 ? "✓" : "✗ 错误!",
           2.0 * N * N * N / 1e9 / (ms / 1000));

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_C);
    return 0;
}

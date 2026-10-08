// matmul_tiled.cu — 分块(tiled)矩阵乘法:共享内存的经典应用
//
// 学习点:
//   1. naive matmul 的问题:C[i][j] 累加时每读 A/B 一行/列都要走全局内存,
//      每个 A 元素被重复读 N 次
//   2. 分块:一个 block 负责 C 的一个 TILE x TILE 子矩阵,
//      A/B 的对应 tile 先载入共享内存,block 内线程复用
//   3. __shared__ + __syncthreads() 的协作模式
//
// 编译运行: make run T=matmul_tiled
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); exit(1); } \
} while (0)

#define N 1024              // 矩阵边长
#define TILE 32             // tile 边长 = block 尺寸

// ---- naive 版:每线程直接从全局内存读 A 的行和 B 的列 ----
__global__ void matmul_naive(const float *A, const float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float acc = 0.0f;
        for (int k = 0; k < n; ++k)
            acc += A[row * n + k] * B[k * n + col];
        C[row * n + col] = acc;
    }
}

// ---- tiled 版:A、B 的 tile 先进共享内存,block 内复用 ----
// 注意两个数组都声明为 [TILE][TILE+1]:
//   sA[ty][k]  warp 内所有线程同地址 -> 广播,无冲突
//   sB[k][tx]  warp 内线程跨行访问同列 -> 无 padding 时是 32 路 bank conflict!
__global__ void matmul_tiled(const float *A, const float *B, float *C, int n) {
    __shared__ float sA[TILE][TILE + 1];
    __shared__ float sB[TILE][TILE + 1];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    // 沿 K 维逐 tile 推进
    for (int t = 0; t < n / TILE; ++t) {
        // 协作载入:每线程各搬一个元素进共享内存
        sA[threadIdx.y][threadIdx.x] = A[row * n + (t * TILE + threadIdx.x)];
        sB[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * n + col];
        __syncthreads();               // 等 tile 载入完成

        // 在共享内存上做 TILE 次乘加:数据从共享内存读,不再走显存
        for (int k = 0; k < TILE; ++k)
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();               // 进入下个 tile 前确保无人还在读
    }

    if (row < n && col < n) C[row * n + col] = acc;
}

void run(void (*launch)(const float *, const float *, float *, dim3, dim3),
         const char *name, const float *dA, const float *dB, float *dC,
         float *hC, dim3 grid, dim3 block) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaMemset(dC, 0, sizeof(float) * N * N);
    cudaEventRecord(start);
    launch(dA, dB, dC, grid, block);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms; cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(hC, dC, sizeof(float) * N * N, cudaMemcpyDeviceToHost);
    int errors = 0;
    for (int i = 0; i < N * N && errors == 0; ++i)
        if (hC[i] != (float)N) { errors++; break; }
    printf("%-12s: %8.2f ms  %s (eff %.1f GFLOPS)\n",
           name, ms, errors == 0 ? "OK" : "FAIL",
           2.0 * N * N * N / 1e9 / (ms / 1000));
}

static void launch_naive(const float *A, const float *B, float *C, dim3 grid, dim3 block) {
    matmul_naive<<<grid, block>>>(A, B, C, N);
}
static void launch_tiled(const float *A, const float *B, float *C, dim3 grid, dim3 block) {
    matmul_tiled<<<grid, block>>>(A, B, C, N);
}

int main() {
    const size_t bytes = sizeof(float) * N * N;
    float *h_A = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);
    for (int i = 0; i < N * N; ++i) h_A[i] = 1.0f;   // A、B 全 1,则 C 每项 = N

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, bytes);
    cudaMalloc((void **)&d_B, bytes);
    cudaMalloc((void **)&d_C, bytes);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_A, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid(N / TILE, N / TILE);
    run(launch_naive, "naive", d_A, d_B, d_C, h_C, grid, block);
    run(launch_tiled, "tiled", d_A, d_B, d_C, h_C, grid, block);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_C);
    return 0;
}

// transpose_tiled.cu — 练习参考解答:分块矩阵转置(带 padding 消 bank conflict)
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); exit(1); } \
} while (0)

#define N 2048
#define TILE 32

__global__ void transpose_tiled(const float *in, float *out, int n) {
    __shared__ float tile[TILE][TILE + 1];   // padding 消除写共享内存时的 bank conflict

    // 读:全局下标按 (row, col) 对齐,读合并
    int col = blockIdx.x * TILE + threadIdx.x;
    int row = blockIdx.y * TILE + threadIdx.y;
    tile[threadIdx.y][threadIdx.x] = in[row * n + col];
    __syncthreads();

    // 写:转置后的全局坐标 —— 同一 warp 的 32 线程写出的是同一"行",写合并
    int out_col = blockIdx.y * TILE + threadIdx.x;
    int out_row = blockIdx.x * TILE + threadIdx.y;
    out[out_row * n + out_col] = tile[threadIdx.x][threadIdx.y];
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

    dim3 block(TILE, TILE);
    dim3 grid(N / TILE, N / TILE);
    transpose_tiled<<<grid, block>>>(d_in, d_out, N);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int r = 0; r < N && errors == 0; ++r)
        for (int c = 0; c < N; ++c)
            if (h_out[c * N + r] != h_in[r * N + c]) { errors++; break; }
    printf("transpose_tiled %dx%d: %s\n", N, N, errors == 0 ? "all elements OK" : "FAILED");

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);
    return 0;
}

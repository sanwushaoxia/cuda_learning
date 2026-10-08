// transpose_naive.cu — 练习参考解答:矩阵转置(naive 版)
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); exit(1); } \
} while (0)

#define M 2048
#define K 2048

__global__ void transpose_naive(const float *in, float *out, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // 输入的列
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // 输入的行
    if (row < rows && col < cols) {
        out[col * rows + row] = in[row * cols + col];
    }
}

int main() {
    const size_t bytes = sizeof(float) * M * K;

    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < K; ++c)
            h_in[r * K + c] = (float)(r * K + c);

    float *d_in, *d_out;
    CHECK(cudaMalloc((void **)&d_in, bytes));
    CHECK(cudaMalloc((void **)&d_out, bytes));
    CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    dim3 block(32, 32);
    dim3 grid((K + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    transpose_naive<<<grid, block>>>(d_in, d_out, M, K);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int r = 0; r < M && errors == 0; ++r)
        for (int c = 0; c < K; ++c)
            if (h_out[c * M + r] != h_in[r * K + c]) { errors++; break; }
    printf("transpose_naive %dx%d: %s\n", M, K, errors == 0 ? "全部元素正确 ✓" : "存在错误!");

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);
    return 0;
}

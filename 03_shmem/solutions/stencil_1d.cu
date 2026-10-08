// stencil_1d.cu — 练习参考解答:一维 3 点 stencil
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); exit(1); } \
} while (0)

#define BLOCK_SIZE 256
#define RADIUS 1

__global__ void stencil_1d(const float *in, float *out, int n) {
    __shared__ float temp[BLOCK_SIZE + 2 * RADIUS];

    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    int lidx = threadIdx.x + RADIUS;

    // 搬主体
    temp[lidx] = (gidx < n) ? in[gidx] : 0.0f;
    // 边界线程搬 halo(越界的全局位置按 0 处理)
    if (threadIdx.x < RADIUS) {
        int left = gidx - RADIUS;
        temp[threadIdx.x] = (left >= 0) ? in[left] : 0.0f;
        int right = gidx + BLOCK_SIZE;
        temp[lidx + BLOCK_SIZE] = (right < n) ? in[right] : 0.0f;
    }
    __syncthreads();

    if (gidx < n) {
        float acc = 0.0f;
        for (int d = -RADIUS; d <= RADIUS; ++d)
            acc += temp[lidx + d];
        out[gidx] = acc;
    }
}

int main() {
    const int N = 1 << 20;
    const size_t bytes = sizeof(float) * N;
    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    for (int i = 0; i < N; ++i) h_in[i] = 1.0f;

    float *d_in, *d_out;
    CHECK(cudaMalloc((void **)&d_in, bytes));
    CHECK(cudaMalloc((void **)&d_out, bytes));
    CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    const int block_size = BLOCK_SIZE;
    const int grid_size = (N + block_size - 1) / block_size;
    stencil_1d<<<grid_size, block_size>>>(d_in, d_out, N);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    // 内部元素应为 3,边界保持 1(边界按不变处理,这里全 1 无法区分,
    // 若 h_in 非全 1,校验应跳过前 RADIUS 和后 RADIUS 个)
    int errors = 0;
    for (int i = RADIUS; i < N - RADIUS; ++i)
        if (h_out[i] != 3.0f) errors++;
    printf("stencil_1d N=%d: 内部元素 %s\n", N, errors == 0 ? "全部 = 3 ✓" : "存在错误!");

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);
    return 0;
}

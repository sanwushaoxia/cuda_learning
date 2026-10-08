// stencil_1d.cu — 练习:一维 stencil(模板计算)
//
// stencil 是很多数值计算(热传导、卷积、图像模糊)的抽象:
//   out[i] = in[i-1] + in[i] + in[i+1]   (3 点 stencil)
//
// 要求:
//   1. 每个 block 把自己负责的范围(含左右各 1 的 halo)载入 __shared__
//   2. 验证 halo 区域:边界 block 的邻居元素来自全局内存
//   3. 边界元素保持不变:out[0] = in[0], out[n-1] = in[n-1]
//   4. 校验正确性
//
// 完成后运行: make run T=stencil_1d
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); exit(1); } \
} while (0)

#define RADIUS 1

// TODO: 实现 3 点 stencil kernel
// 提示:
//   __shared__ float temp[BLOCK_SIZE + 2 * RADIUS];
//   全局下标 gidx = blockIdx.x * blockDim.x + threadIdx.x;
//   temp[RADIUS + threadIdx.x] = in[gidx];
//   边界线程再负责搬 halo: temp[0] / temp[BLOCK_SIZE+1]
//   __syncthreads() 后: out[gidx] = temp[..-1] + temp[..] + temp[..+1]
__global__ void stencil_1d(const float *in, float *out, int n) {
    // 在此写代码
}

int main() {
    const int N = 1 << 20;
    const size_t bytes = sizeof(float) * N;
    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    for (int i = 0; i < N; ++i) h_in[i] = 1.0f;      // 全 1,stencil 结果也是 1

    float *d_in, *d_out;
    CHECK(cudaMalloc((void **)&d_in, bytes));
    CHECK(cudaMalloc((void **)&d_out, bytes));
    CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // TODO: 选择 block_size=256,启动 kernel,拷回并校验

    printf("stencil_1d: 尚未实现\n");
    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);
    return 0;
}

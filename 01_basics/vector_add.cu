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

// 方式一:一线程一元素,需保证 grid 覆盖全部 N
__global__ void vector_add_one_per_thread(const float *x, const float *y, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {                      // 最后一个 block 可能有越界线程,必须判断
        out[i] = x[i] + y[i];
    }
}

// 方式二:网格跨越循环(推荐惯用法,对任意 N 都正确)
__global__ void vector_add_grid_stride(const float *x, const float *y, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;   // 线程总数
    for (; i < n; i += stride) {
        out[i] = x[i] + y[i];
    }
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

    // 3. 启动 kernel: 256 线程/block,向上取整算 block 数
    const int block_size = 256;
    const int grid_size = (N + block_size - 1) / block_size;
    vector_add_one_per_thread<<<grid_size, block_size>>>(d_x, d_y, d_out, N);

    // 4. 取回结果(kernel 启动是异步的,cudaMemcpy D2H 会隐式同步)
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // 5. 验证
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        if (h_out[i] != 3.0f) { errors++; }
    }
    printf("N=%d, %s\n", N, errors == 0 ? "全部元素正确 ✓" : "存在错误!");

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_out);
    free(h_x); free(h_y); free(h_out);
    return 0;
}

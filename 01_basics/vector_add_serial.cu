// vector_add_serial.cu — 对照:与原 vector_add.cu 相同的"单线程 kernel"写法
//
// 保留这份代码的目的:它是初学者最常犯的错误 —— 把 CPU 的 for 循环
// 原封不动搬进 kernel,只用 <<<1,1>>> 启动。这样 GPU 上只有 1 个线程
// 在串行执行,加上数据来回拷贝,比纯 CPU 还慢。
//
// 对比运行: make run T=vector_add_serial && make run T=vector_add
#include <stdio.h>
#include <stdlib.h>
#include "../cuda_common.cuh"

// 反面教材:单线程 kernel 串行 for 循环
__global__ void add_serial(const float *x, const float *y, float *out, int n) {
    for (int i = 0; i < n; ++i) {
        out[i] = x[i] + y[i];
    }
}

int main() {
    const int N = 1 << 24;
    const size_t bytes = sizeof(float) * N;

    float *h_x = (float *)malloc(bytes);
    float *h_y = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    for (int i = 0; i < N; ++i) {
        h_x[i] = 1.0f;
        h_y[i] = 2.0f;
    }

    float *d_x, *d_y, *d_out;
    cudaMalloc((void **)&d_x, bytes);
    cudaMalloc((void **)&d_y, bytes);
    cudaMalloc((void **)&d_out, bytes);
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice);

    // 仅计时 kernel(与 vector_add 对比;不含 H2D/D2H)
    // 注意:语句含逗号时必须整体加括号,否则被当成宏的多个参数
    float ms;
    CUDA_TIME((add_serial<<<1, 1>>>(d_x, d_y, d_out, N)), ms);

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    printf("vector_add serial 1-thread (N=%d): %.2f ms -- correct but not parallel\n", N, ms);

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_out);
    free(h_x); free(h_y); free(h_out);
    return 0;
}

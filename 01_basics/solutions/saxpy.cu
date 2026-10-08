// saxpy.cu — 练习 1 参考解答
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void saxpy_kernel(float a, const float *x, const float *y, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a * x[i] + y[i];
    }
}

int main() {
    const int N = 1 << 20;
    const float a = 2.5f;
    const size_t bytes = sizeof(float) * N;

    float *h_x = (float *)malloc(bytes);
    float *h_y = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    for (int i = 0; i < N; ++i) {
        h_x[i] = (float)i;
        h_y[i] = 1.0f;
    }

    float *d_x, *d_y, *d_out;
    cudaMalloc((void **)&d_x, bytes);
    cudaMalloc((void **)&d_y, bytes);
    cudaMalloc((void **)&d_out, bytes);
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice);

    const int block_size = 256;
    const int grid_size = (N + block_size - 1) / block_size;
    saxpy_kernel<<<grid_size, block_size>>>(a, d_x, d_y, d_out, N);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    int errors = 0;
    for (int i = 0; i < N; ++i) {
        if (h_out[i] != a * (float)i + 1.0f) { errors++; }
    }
    printf("saxpy N=%d: %s\n", N, errors == 0 ? "all elements OK" : "FAILED");

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_out);
    free(h_x); free(h_y); free(h_out);
    return 0;
}

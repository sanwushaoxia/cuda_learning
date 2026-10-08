// saxpy.cu — 练习 1:实现 SAXPY
//
// SAXPY 是 BLAS 中的经典操作: out[i] = a * x[i] + y[i]
// 它与 vector_add 只差一个系数乘法,重点练习:
//   1. 带标量参数的 kernel(标量按值传递即可,不必放显存)
//   2. 完整的分配 -> 拷贝 -> 启动 -> 取回 -> 释放流程
//   3. 用 CUDA_CHECK 检查每个 API 调用
//
// 完成后运行: make run T=saxpy
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// TODO: 实现 kernel。提示: 与 vector_add_one_per_thread 几乎相同,多乘一个 a
__global__ void saxpy_kernel(float a, const float *x, const float *y, float *out, int n) {
    // 在此写代码
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

    // TODO: 分配 device 内存 (cudaMalloc)
    // TODO: 拷贝 h_x/h_y 到 device (cudaMemcpy)
    // TODO: 选择 block_size=256,计算 grid_size 并启动 kernel
    // TODO: 取回 h_out,同步
    // TODO: 验证: h_out[i] 应等于 a * i + 1.0f
    // TODO: 释放内存

    (void)a; (void)h_out;   // 占位,完成后删除
    printf("saxpy: 尚未实现\n");
    free(h_x); free(h_y); free(h_out);
    return 0;
}

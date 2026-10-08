// reduce_warp.cu — warp shuffle 归约:避免共享内存同步的终极优化
//
// 学习点:
//   1. warp 是 GPU 的最小调度单位(32 线程),warp 内天然锁步执行
//   2. __shfl_down_sync 在寄存器间直接交换数据,不需要共享内存和 __syncthreads
//   3. 层次化归约:先 warp 内 shuffle 归约(无同步),每 warp 再写共享内存,
//      最后一个 warp 完成收尾 —— 1 个 block 只需 1 次跨 warp 同步
//   4. volatile / __syncwarp 的坑:跨 warp 读共享内存必须 volatile
//
// 编译运行: make run T=reduce_warp
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); exit(1); } \
} while (0)

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// warp 内归约:5 步 shuffle,sum 全 warp 每个线程都有
__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// block 内归约:shuffle(无同步) + 一次共享内存跨 warp 汇合
__global__ void reduce_warp(const float *in, float *out, int n) {
    __shared__ float warp_sums[WARP_SIZE];

    // 网格跨越:每线程累计多个元素,减少 block 数与启动开销
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * blockDim.x)
        sum += in[i];

    sum = warp_reduce_sum(sum);           // warp 内归约

    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    if (lane == 0) warp_sums[warp_id] = sum;   // 每 warp 的 lane 0 写共享内存
    __syncthreads();                           // 唯一一次 block 级同步

    // 用前几个线程再归约各 warp 的部分和(此时 warp_sums 有 block 内 warps 个有效值)
    if (warp_id == 0) {
        sum = (lane < blockDim.x / WARP_SIZE) ? warp_sums[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane == 0) atomicAdd(out, sum);    // 跨 block 用原子加
    }
}

int main() {
    const int N = 1 << 26;                    // 6700 万元素
    const size_t bytes = sizeof(float) * N;
    float *h_in = (float *)malloc(bytes);
    for (int i = 0; i < N; ++i) h_in[i] = 1.0f;

    float *d_in, *d_out;
    CHECK(cudaMalloc((void **)&d_in, bytes));
    CHECK(cudaMalloc((void **)&d_out, sizeof(float)));
    CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // 固定 block 数,靠 grid-stride 覆盖全部数据(利于重复测量稳定)
    const int blocks = 2048;
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // 预热
    CHECK(cudaMemset(d_out, 0, sizeof(float)));
    reduce_warp<<<blocks, BLOCK_SIZE>>>(d_in, d_out, N);
    CHECK(cudaDeviceSynchronize());

    float total_ms = 0;
    const int iters = 20;
    for (int it = 0; it < iters; ++it) {
        CHECK(cudaMemset(d_out, 0, sizeof(float)));
        cudaEventRecord(start);
        reduce_warp<<<blocks, BLOCK_SIZE>>>(d_in, d_out, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }
    float h_out;
    CHECK(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    printf("N=%d, expected sum %d\n", N, N);
    printf("warp shuffle reduce: %.0f  %s\n", h_out, h_out == (float)N ? "OK" : "FAIL");
    printf("avg %.3f ms/iter (eff BW %.0f GB/s)\n",
           total_ms / iters, bytes / 1e9 / (total_ms / iters / 1000));

    cudaFree(d_in); cudaFree(d_out); free(h_in);
    return 0;
}

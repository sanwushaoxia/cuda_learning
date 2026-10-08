// cuda_graph.cu — CUDA Graph:降低重复执行的启动开销
//
// 学习点:
//   1. 每个 kernel launch 有 CPU 端 5~10us 开销;上千个小 kernel 串成
//      工作流时,launch 开销成为瓶颈(推理框架的常见问题)
//   2. CUDA Graph 把一串操作"录制"成图,之后一次 launch 全部重放
//   3. 适合:固定结构的重复工作流(每帧/每步相同的 kernel 序列)
//
// 编译运行: make run T=cuda_graph
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); exit(1); } \
} while (0)

#define N (1 << 20)
#define STAGES 64            // 模拟推理的多阶段流水:64 个小 kernel

__global__ void stage_kernel(float *data, int n, float f) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = fmaf(data[i], f, 0.1f);
}

int main() {
    const size_t bytes = sizeof(float) * N;
    float *d_data;
    CHECK(cudaMalloc((void **)&d_data, bytes));
    CHECK(cudaMemset(d_data, 1, bytes));

    const int grid = (N + 255) / 256, block = 256;
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // ---- 方式一:朴素重复 launch ----
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int it = 0; it < 10; ++it)
        for (int s = 0; s < STAGES; ++s)
            stage_kernel<<<grid, block>>>(d_data, N, 1.001f);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float naive_ms;
    cudaEventElapsedTime(&naive_ms, start, stop);

    // ---- 方式二:录制 Graph,一次 launch 重放 ----
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (int s = 0; s < STAGES; ++s)
        stage_kernel<<<grid, block, 0, stream>>>(d_data, N, 1.001f);
    CHECK(cudaStreamEndCapture(stream, &graph));
    CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int it = 0; it < 10; ++it)
        CHECK(cudaGraphLaunch(graph_exec, stream));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float graph_ms;
    cudaEventElapsedTime(&graph_ms, start, stop);

    printf("%d small kernels/run, replay x10:\n", STAGES);
    printf("  naive per-launch : %7.2f ms\n", naive_ms);
    printf("  CUDA Graph      : %7.2f ms  (saved %0.f%% launch overhead)\n",
           graph_ms, 100 * (1 - graph_ms / naive_ms));

    CHECK(cudaGraphExecDestroy(graph_exec));
    CHECK(cudaGraphDestroy(graph));
    CHECK(cudaStreamDestroy(stream));
    cudaFree(d_data);
    return 0;
}

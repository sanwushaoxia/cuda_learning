// cuda_common.cuh — 各阶段示例共用的错误检查宏与计时工具
//
// CUDA 运行时 API 的错误是"静默"的: kernel 启动失败不会抛异常,
// 错误码需要主动检查。工程惯例是用一个宏包裹所有 API 调用。
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err__));          \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// kernel 启动后的检查(能捕获 launch 配置错误,如线程数超限)
#define CUDA_CHECK_LAUNCH()                                                  \
    do {                                                                     \
        cudaError_t err__ = cudaGetLastError();                              \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "Kernel launch failed at %s:%d: %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(err__));          \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
        CUDA_CHECK(cudaDeviceSynchronize());                                 \
    } while (0)

// ---- 计时工具 ------------------------------------------------------------
// kernel 是异步的,host 的 clock 测不到执行耗时;必须用 cudaEvent,
// 且读耗时要先同步(见 01_basics/README.md「观测方法」)。

// CUDA_TIME(stmt, ms_var):用 event 包住任意一条(段)GPU 语句,
// 把纯 GPU 执行耗时(毫秒)写入 ms_var(float)。
// 用法:  float ms; CUDA_TIME((kernel<<<g,b>>>(args)), ms);
#define CUDA_TIME(stmt, ms_var)                                              \
    do {                                                                     \
        cudaEvent_t start__, stop__;                                         \
        CUDA_CHECK(cudaEventCreate(&start__));                               \
        CUDA_CHECK(cudaEventCreate(&stop__));                                \
        CUDA_CHECK(cudaEventRecord(start__));                                \
        stmt;                                                                \
        CUDA_CHECK(cudaEventRecord(stop__));                                 \
        CUDA_CHECK(cudaEventSynchronize(stop__));                            \
        CUDA_CHECK(cudaEventElapsedTime(&(ms_var), start__, stop__));        \
        CUDA_CHECK(cudaEventDestroy(start__));                               \
        CUDA_CHECK(cudaEventDestroy(stop__));                                \
    } while (0)

// CUDA_TIME_AVG(stmt, ms_var, iters):预热 1 次后测 iters 次取平均,
// 消除首次启动的 context/模块加载开销与单次噪声(±5%)。
// 注意:stmt 会被执行 iters+1 次,须保证重复执行无副作用。
#define CUDA_TIME_AVG(stmt, ms_var, iters)                                   \
    do {                                                                     \
        float once__;                                                        \
        CUDA_TIME(stmt, once__);                 /* 预热 */                   \
        (ms_var) = 0.0f;                                                     \
        for (int i__ = 0; i__ < (iters); ++i__) {                            \
            CUDA_TIME(stmt, once__);                                         \
            (ms_var) += once__;                                              \
        }                                                                    \
        (ms_var) /= (iters);                                                 \
    } while (0)


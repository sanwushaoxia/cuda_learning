// cuda_common.cuh — 各阶段示例共用的错误检查宏
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

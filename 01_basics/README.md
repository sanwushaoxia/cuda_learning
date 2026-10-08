# 阶段一:编程模型与起步(初学者)

目标:理解 CUDA 的线程层次模型,能独立写出、编译并运行一个正确的 kernel。

前置要求:会用命令行,有 C/C++ 基础(指针、数组、动态内存)。

预计用时:2~4 小时。

## 核心概念

### 1. 线程层次:thread → block → grid

CUDA 把并行工作组织成三级:

- **thread**:最小执行单元,每个线程执行一遍 kernel 代码;
- **block**:一组线程,可通过共享内存通信、用 `__syncthreads()` 同步;
- **grid**:一次 kernel 启动的所有 block。

每个线程有唯一的内置坐标,据此算出自己负责的数据下标:

```c
int i = blockIdx.x * blockDim.x + threadIdx.x;   // 一维全局索引
```

二维/三维问题(矩阵、图像)用 `blockIdx.y`、`threadIdx.y` 同理扩展。

### 2. kernel 启动语法

```c
kernel<<<grid_size, block_size>>>(参数...);
```

- `block_size`:每 block 线程数,通常是 128/256/512(须为 32 的倍数);
- `grid_size`:block 数量,惯用向上取整 `(N + block_size - 1) / block_size`;
- kernel 启动是**异步**的:CPU 立即返回,需要 `cudaDeviceSynchronize()` 或后续
  `cudaMemcpy` 隐式等待结果。

### 3. 错误检查

CUDA 的错误是静默的,不会抛异常。请从第一天起使用 `cuda_common.cuh` 里的
`CUDA_CHECK(...)` 包裹每个 API 调用,并在 kernel 启动后调用 `CUDA_CHECK_LAUNCH()`。

## 文件说明

| 文件 | 说明 | 运行命令 |
|------|------|----------|
| `hello_world.cu` | 最简 kernel,打印线程编号 | `make run T=hello_world` |
| `vector_add_serial.cu` | 反面教材:单线程 kernel 串行循环,附计时 | `make run T=vector_add_serial` |
| `vector_add.cu` | 正确写法:一线程一元素 + grid-stride 循环 | `make run T=vector_add` |
| `device_info.cu` | 查询 GPU 属性(SM 数、最大线程数、显存等) | `make run T=device_info` |
| `cuda_common.cuh` | 共用错误检查宏 `CUDA_CHECK` | 被其他文件 include |

## 建议阅读顺序

1. `hello_world.cu` — 感受 kernel 语法与同步;
2. `vector_add_serial.cu` — 跑一遍,记住这个反面教材;
3. `vector_add.cu` — 对照理解"线程负责元素"的思维方式;
4. `device_info.cu` — 认识你的 GPU(后面优化阶段会反复用到这些参数)。

## 练习

| 练习 | 说明 | 解答 |
|------|------|------|
| `exercises/saxpy.cu` | 实现 `out[i] = a*x[i] + y[i]`,完整走一遍流程 | `solutions/saxpy.cu` |
| `exercises/thread_grid_2d.cu` | 二维 grid 打印线程坐标,理解全局索引 | `solutions/thread_grid_2d.cu` |

先只看练习文件里的 TODO 自己写,写完编译运行通过后再对照解答。

## 自检问题

- 为什么 kernel 里必须有 `if (i < n)` 判断?去掉会怎样?
- `<<<1, 1>>>` 和 `<<<N/256, 256>>>` 的区别是什么?
- kernel 启动后不 `cudaDeviceSynchronize()`,直接 printf host 数组会发生什么?

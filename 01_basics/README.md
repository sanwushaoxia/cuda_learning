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

二维问题(矩阵、图像)同理扩展到 `(row, col)`:

```c
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
// 行主序一维下标: idx = row * cols + col
```

约定:通常 **x → 列(宽)、y → 行(高)**。

### 2. kernel 启动语法

一维(向量):

```c
kernel<<<grid_size, block_size>>>(参数...);
// block_size: 每 block 线程数,常用 128/256/512(宜为 32 的倍数)
// grid_size : block 数,向上取整 (N + block_size - 1) / block_size
```

二维用 `dim3` 同时指定 x/y(未写的分量默认为 1):

```c
dim3 block(2, 2);   // blockDim.x=2, blockDim.y=2 → 每 block 2×2=4 线程
dim3 grid((cols + block.x - 1) / block.x,
          (rows + block.y - 1) / block.y);  // 覆盖 rows×cols
kernel<<<grid, block>>>(rows, cols);
```

例:`rows=cols=8`、`block(2,2)` → `grid` 为 `(4,4)`,共 16 个 block、64 个线程,
正好一线程对应一个 `(row,col)`。启动后仍要边界判断 `row < rows && col < cols`
(grid 向上取整时边缘 block 可能越界)。

对照 `exercises/thread_grid_2d.cu` 练习二维坐标打印。

kernel 启动是**异步**的:CPU 立即返回,需要 `cudaDeviceSynchronize()` 或后续
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
- `dim3 block(2,2)` 与 `dim3 grid(4,4)` 时,全局 `(row,col)` 怎么算?总共多少线程?

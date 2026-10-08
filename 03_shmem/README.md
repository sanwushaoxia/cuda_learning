# 阶段三:共享内存与线程协作(进阶)

目标:掌握 block 内线程通过共享内存协作的经典模式 —— 分块(tiling)、
归约、halo 交换,并理解 bank conflict。

前置要求:完成阶段二。

预计用时:4~6 小时。

## 核心概念

### 1. 共享内存与同步

- `__shared__` 声明 block 内共享的片上内存,延迟接近寄存器;
- 共享内存对 block 内所有线程可见,但**没有自动顺序保证** ——
  跨线程读写必须用 `__syncthreads()` 隔开;
- 容量有限:本机每 block 48KB(可配置更大),对应阶段一的 `device_info` 输出。

### 2. 分块 (tiling) 模式

全局内存带宽远低于共享内存。让 block 先把要复用的数据"搬进"共享内存,
所有线程再从共享内存读,这就是分块。`matmul_tiled.cu` 展示了标准套路:

```
载入 tile -> __syncthreads() -> 在 tile 上计算 -> __syncthreads() -> 下一 tile
```

### 3. 归约 (reduction)

1M 个数求和,树形两两合并只需 log2(N) 轮。两种经典实现对比:

- 相邻配对:`tid % (2*stride) == 0` 的线程工作 —— warp 内大量分支空闲;
- 交错配对:前半与后半配对 —— 无分支浪费,活跃线程连续,是标准写法。

### 4. bank conflict

共享内存分 32 个 bank,同一 warp 的线程访问同 bank 不同地址则串行化。
经典解法是 **padding**:声明 `[32][33]` 而不是 `[32][32]`,让相邻行错开。

## 文件说明

| 文件 | 说明 | 运行命令 |
|------|------|----------|
| `matmul_tiled.cu` | naive vs tiled 矩阵乘,含 GFLOPS 输出 | `make run T=matmul_tiled` |
| `reduce_basic.cu` | 两种共享内存归约实现 | `make run T=reduce_basic` |

## 练习

| 练习 | 说明 | 解答 |
|------|------|------|
| `exercises/transpose_tiled.cu` | 分块转置 + padding 消 bank conflict | `solutions/transpose_tiled.cu` |
| `exercises/stencil_1d.cu` | 带 halo 的一维 3 点模板计算 | `solutions/stencil_1d.cu` |

## 自检问题

- tiled matmul 里两个 `__syncthreads()` 各自防止什么错误?
- 为什么交错配对归约比相邻配对快?画出 warp 内线程的活跃情况。
- `[32][33]` 的 padding 为什么能消除 bank conflict?
- stencil 的 halo 是什么?不加 halo 会算错哪些元素?

# 阶段二:内存层次与传输(初学者→进阶)

目标:掌握显存管理生命周期,理解 host/device 传输是很多程序的第一瓶颈,
并学会用 cudaEvent 给 GPU 侧操作计时。

前置要求:完成阶段一。

预计用时:3~5 小时。

## 核心概念

### 1. 内存层次

| 类型 | 位置 | 作用域 | 特点 |
|------|------|--------|------|
| 全局内存 (global) | 显存(GDDR6) | 所有线程 | 大(12GB)但慢,~360 GB/s |
| 共享内存 (shared) | SM 片上 | 同一 block | 小(48KB/block)但极快,类 L1 |
| 常量内存 (constant) | 显存缓存 | 所有线程 | 只读,广播访问高效 |
| 寄存器 | SM | 单线程 | 最快,自动分配 |

#### 共享内存用法要点

- **作用域**:`__shared__` 仅同一 block 内线程可见、共享;不同 block 各有一份,互不可见。
- **容量上限**:由设备决定(见 `sharedMemPerBlock`,本机常见 ~48KB/block),
  **与 `blockDim` 无关** —— 线程开得再多,也不会自动多给共享内存。
- **实际用量**:由你声明决定,例如 `__shared__ float s[256];` 或动态
  `extern __shared__ float s[];` + 启动时第三个配置参数指定字节数。
- **和 block 大小的关系**:常按“每线程一槽”自行对齐(如 256 线程配 `s[256]`),
  这是编程约定;同一 SM 上多 block 时,每 block 占用的共享内存会一起挤占用率。
- **同步**:block 内协作读写前/后用 `__syncthreads()`,否则有数据竞争。
- 典型用途:tile 缓存、block 内归约、消除重复全局内存读。深入见阶段三 `03_shmem`。

```c
__global__ void demo(const float *in, float *out) {
    __shared__ float tile[256];           // 每 block 一份,生命周期随 block
    int t = threadIdx.x;
    tile[t] = in[blockIdx.x * blockDim.x + t];
    __syncthreads();                      // 等全员写完再读
    out[...] = tile[t] + tile[(t + 1) % 256];
}
```

### 2. host ↔ device 传输

- `cudaMalloc/cudaMemcpy/cudaFree` 是手动管理三件套;
- RTX 3060 显存带宽约 360 GB/s,而 PCIe 4.0 x4 只有约 8 GB/s ——
  **数据来回拷贝往往比 kernel 计算本身还贵**;
- `malloc` 的内存可换页,DMA 前要先驱动锁定;`cudaMallocHost`(pinned)
  跳过这一步,带宽通常提升 1.5~2 倍,但占用不可换页的物理内存;
- `cudaMallocManaged` 统一内存:代码最简单,页按需自动迁移,
  适合原型和复杂访问模式,但性能需要用 `cudaMemPrefetchAsync` 调优。

### 3. cudaEvent 计时

`host` 侧 `clock()` 对异步的 CUDA 调用计时不准,正确姿势:

```c
cudaEventRecord(start);
/* GPU 操作 */
cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&ms, start, stop);
```

## 文件说明

| 文件 | 说明 | 运行命令 |
|------|------|----------|
| `mem_transfer.cu` | 测 H2D/D2H 带宽,与设备内 memset 对照 | `make run T=mem_transfer` |
| `pinned_memory.cu` | pageable vs pinned 传输带宽对比 | `make run T=pinned_memory` |
| `unified_memory.cu` | cudaMallocManaged + prefetch 用法 | `make run T=unified_memory` |

## 练习

| 练习 | 说明 | 解答 |
|------|------|------|
| `exercises/transpose_naive.cu` | 二维 grid 实现矩阵转置(阶段三优化它) | `solutions/transpose_naive.cu` |

## 自检问题

- 为什么 D2H 通常比 H2D 慢?(提示:写回 + 页表,可实测验证)
- pinned 内存为什么快?为什么不能全用 pinned?
- 统一内存的 kernel 性能为什么常常不如手动 cudaMemcpy?缺页发生在哪?
- 共享内存作用域是 block;把 block 从 128 改成 1024,每 block 可用共享内存上限变吗?

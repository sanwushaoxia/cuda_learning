# 阶段五:流、并发与 CUDA Graph(专家)

目标:掌握用流重叠传输与计算、多流并发执行、event 表达跨流依赖,
并了解 CUDA Graph 消除重复 launch 开销的现代做法。

前置要求:完成阶段四(会用 pinned 内存与 cudaEvent 计时)。

预计用时:4~6 小时。

## 核心概念

### 1. 流 (stream) 语义

- 同一 stream 内操作保序,不同 stream 之间无序(可并发);
- kernel、`cudaMemcpyAsync` 都接受流参数;同步拷贝(`cudaMemcpy`)不指定流;
- **默认流陷阱**:legacy default stream 与所有流隐式互斥,混用会串行化一切;
- 异步传输生效的前提是 **pinned host 内存**,否则驱动静默退化为同步。

### 2. 流水线模式

把数据分成 N 块,每块"传输 → 计算 → 传回"放入同一流:

```
流0: H2D块0 → kernel块0 → D2H块0
流1:        H2D块1 → kernel块1 → D2H块1
```

总耗时从"三次操作之和"降为"约 max(传输, 计算) + 一块的开销"。

### 3. Event:跨流依赖与计时

```c
cudaEventRecord(ev, s1);          // s1 内打点
cudaStreamWaitEvent(s2, ev);      // s2 等待该点完成
```

比 `cudaStreamSynchronize`(整机等待)更细粒度,能表达 DAG 依赖。

### 4. CUDA Graph

每个 kernel launch 有 CPU 端 5~10us 开销。固定结构的重复工作流
(如推理的每步)录制一次、重放多次,launch 开销几乎归零。

## 文件说明

| 文件 | 说明 | 运行命令 |
|------|------|----------|
| `stream_overlap.cu` | 1/2/4 流的传输-计算流水线对比 | `make run T=stream_overlap` |
| `multi_stream.cu` | 多流并发 kernel + 跨流 event 依赖 | `make run T=multi_stream` |
| `cuda_graph.cu` | 64 kernel 序列:朴素 vs Graph 重放 | `make run T=cuda_graph` |

## 练习

| 练习 | 说明 | 解答 |
|------|------|------|
| `exercises/vector_add_pipelined.cu` | vector_add 分块 + 双流流水线化 | `solutions/vector_add_pipelined.cu` |

## 本机实测的三个"反直觉"现象(RTX 3060)

1. **大 kernel 双流并发反而更慢**:multi_stream 的 busy_kernel 单个就能
   占满 SM,双流并发 15ms > 串行 11ms。流并发只对"喂不饱 GPU 的小 kernel"有效;
2. **传输流水线收益约 15~20%**:stream_overlap 中 57ms → 45ms,
   不是理想的"max(传输, 计算)",因为 PCIe 与 SM 资源仍互相干扰;
3. **大 kernel 的 Graph 收益很小**:cuda_graph 中 64 个 kernel 每个 7.6us
   以上的 GPU 时间,launch 开销占比低;kernel 越小收益越大(试试把 STAGES
   改成每个 kernel 只处理 1024 元素)。

## 自检问题

- 为什么不换 pinned 内存,`cudaMemcpyAsync` 也"能跑"但完全不重叠?
- 4 个流一定比 2 个流快吗?受什么硬件资源约束?(提示:PCIe 和 SM 都只有一份)
- 什么样的工作流适合 CUDA Graph?参数每步都变时怎么办?(查 `cudaGraphExecKernelNodeSetParams`)

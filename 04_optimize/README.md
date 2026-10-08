# 阶段四:优化与 Profiling(专家核心)

目标:建立"测量驱动优化"的工作流 —— 先用 Nsight 工具定位瓶颈,
再用合并访存、warp 原语、占用率分析等手段针对性优化。

前置要求:完成阶段三,理解共享内存与 bank conflict。

预计用时:6~10 小时。

## 核心概念

### 1. 优化决策:先分类瓶颈

| 瓶颈类型 | 典型特征 | 对策 |
|----------|----------|------|
| 访存带宽 | 计算强度低(每字节数学运算少) | 合并访存、减少数据流量、pinned/流重叠 |
| 计算延迟 | 大量数学运算,寄存器压力大 | 指令级优化、tensor core、减少分支 |
| 占用率低 | 活跃 warp 不足以掩盖延迟 | 调 block 尺寸、减寄存器/共享内存占用 |
| 同步开销 | __syncthreads 密集、warp 发散 | warp shuffle、线程粗化、重构算法 |

### 2. Nsight 工具链

```bash
# Nsight Systems:时间线视角 —— kernel/传输/流的重叠与间隙
make profile T=coalesce                      # 本模块封装了 nsys + stats
nsys profile ./build/04_optimize/coalesce    # 手动完整版
nsys stats <report>.nsys-rep                 # kernel 汇总/传输统计

# Nsight Compute:kernel 内部视角 —— 指标级分析(如已安装)
ncu --set full ./build/04_optimize/coalesce  # 关注 SOL、occupancy、warp 发散
ncu --metrics gld_efficiency,gst_efficiency ...
```

关键指标:`dram__throughput`(带宽利用率)、`sm__throughput`(计算利用率)、
`achieved_occupancy`、`warp_execution_efficiency`。

### 3. 四个必须内化的优化手段

1. **合并访存**:warp 32 线程访问连续地址 → `coalesce.cu` 实测差距;
2. **warp shuffle**:`__shfl_down_sync` 寄存器间交换,归约免共享内存同步;
3. **线程粗化 (coarsening)**:每线程做更多工作,摊薄索引计算与访存;
4. **分支对齐**:让同一 warp 走同一分支,避免发散的"都执行"浪费。

## 文件说明

| 文件 | 说明 | 运行命令 |
|------|------|----------|
| `coalesce.cu` | 合并 vs 跨步访存带宽对比 | `make run T=coalesce` |
| `divergence.cu` | warp 发散 vs 对齐分支耗时对比 | `make run T=divergence` |
| `reduce_warp.cu` | warp shuffle 层次化归约 | `make run T=reduce_warp` |
| `occupancy.cu` | block 尺寸扫描 + 占用率 API | `make run T=occupancy` |

## 练习

| 练习 | 说明 | 解答 |
|------|------|------|
| `exercises/matmul_opt.cu` | 把阶段三 tiled matmul 优化到位并出 profiling 报告 | `solutions/matmul_opt.cu` |

练习要求:每做一步优化都记录耗时与 ncu 指标,写出"改了什么→指标怎么变→为什么"。

## 本机实测参考(RTX 3060, 2026-10)

- 合并访存约 300+ GB/s,跨步访问跌到几十 GB/s(差距 5~10 倍);
- matmul 三级台阶:naive ~200 GFLOPS → tiled ~470 GFLOPS → 2x2 寄存器分块 ~1.1 TFLOPS。

## 自检问题

- 为什么"提高占用率"不是万能药?什么时候低占用率反而更快?
- warp shuffle 为什么不需要 __syncthreads?
- 一个 kernel 带宽利用率只有 20%,你的排查清单是什么?

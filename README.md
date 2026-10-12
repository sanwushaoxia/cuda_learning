# CUDA 学习指导:从初学者到专家

教学向 CUDA 编程学习模块,六个阶段覆盖线程模型 → 内存层次 → 共享内存协作 →
性能优化与 profiling → 流并发 → PyTorch 自定义扩展。每个阶段都有
**讲解 + 可运行示例 + 练习(TODO 骨架)+ 参考解答**。

模块支持**双机双 GPU 配置**(configs/ 下按机器分 profile),默认使用本机
RTX 3060,可切换回 GTX 750 Ti;文末实测数据来自 RTX 3060。

## 环境准备

| 配置项 | rtx3060(**默认**) | gtx750ti |
|--------|--------------------|----------|
| GPU | NVIDIA GeForce RTX 3060 (12GB, Ampere) | NVIDIA GeForce GTX 750 Ti (2GB, Maxwell) |
| 架构 (`-arch`) | `sm_86` | `sm_50` |
| CUDA Toolkit | 12.4 (`/usr/local/cuda`) | 11.8(`C:/Program Files/.../CUDA/v11.8`,sm_50 支持的最后版本) |
| 系统 | Linux | Windows(经 `tools/nvcc_win.cmd` + VS 2019 BuildTools) |
| 产物目录 | `build/rtx3060/` | `build/gtx750ti/` |
| 阶段六(PyTorch 扩展) | ✓(conda `py312`,PyTorch 2.6.0+cu124) | 不适用 |

### 切换配置

```bash
# 方式一:命令行临时切换
make GPU_PROFILE=gtx750ti all
make GPU_PROFILE=gtx750ti run T=vector_add

# 方式二:改 config.mk 里的 GPU_PROFILE 一行,长期生效
```

两个 profile 的编译产物在 `build/<profile>/` 下**互不覆盖**,切换后无需
`make clean`(跨架构切换后建议清理省空间)。profile 文件在 `configs/` 下,
换新机器时复制一份改 `GPU_ARCH` 和 `CUDA_HOME` 即可。

### 环境自检

```bash
nvcc --version              # 确认 toolkit
nvidia-smi                  # 确认驱动与 GPU
conda activate py312        # 仅阶段六需要
```

## 学习路线

| 阶段 | 目录 | 主题 | 关键词 | 预计用时 |
|------|------|------|--------|----------|
| 一 | [01_basics](01_basics/README.md) | 编程模型与起步 | thread/block/grid、启动语法、错误检查 | 2~4h |
| 二 | [02_memory](02_memory/README.md) | 内存层次与传输 | 显存生命周期、pinned、统一内存、cudaEvent 计时 | 3~5h |
| 三 | [03_shmem](03_shmem/README.md) | 共享内存与协作 | tiling、归约、bank conflict、halo | 4~6h |
| 四 | [04_optimize](04_optimize/README.md) | 优化与 profiling(专家核心) | 合并访存、warp shuffle、占用率、nsys/ncu | 6~10h |
| 五 | [05_streams](05_streams/README.md) | 流、并发与 Graph | 流水线、跨流 event、CUDA Graph | 4~6h |
| 六 | [06_pytorch](06_pytorch/README.md) | PyTorch 自定义扩展 | cpp_extension.load、当前流、算子融合 | 3~5h |

学习方式:每阶段先读 README → 跑示例 → 只看 `exercises/` 的 TODO 自己写
→ 通过后对照 `solutions/`。阶段之间强依赖(三依赖二,四依赖三),一、二
可与任何项目并行学习。

## 构建与运行速查

```bash
make all                    # 编译全部示例(01-05 阶段,含练习与解答)
make run T=vector_add       # 运行某个示例(按文件名查找)
make profile T=coalesce     # nsys profiling + kernel/传输统计
make clean                  # 清理当前 profile 的 build/<profile>/

# 切换到 GTX 750 Ti(Windows 机):make GPU_PROFILE=gtx750ti all
# 或长期修改 config.mk 的 GPU_PROFILE 一行

# 阶段六(PyTorch 扩展,仅 rtx3060 profile,由 Python 驱动编译):
conda activate py312
cd 06_pytorch && python extension/demo.py
```

## 本机实测成绩单(RTX 3060, 2026-10,供对照)

> 以下数据来自 RTX 3060 profile。GTX 750 Ti(sm_50, 2GB)平台上部分示例
> 需要调小数据规模,阶段五的 CUDA Graph 等新特性依赖 sm_70+,仅供参考;
> 750Ti 实测数据待补充。

| 项目 | 结果 |
|------|------|
| vector_add 单线程 kernel | 1645 ms(反面教材) |
| PCIe H2D/D2H 带宽 | ~6.9 / 6.6 GB/s |
| matmul 1024³:naive → tiled → 2x2 寄存器分块 | 200 → 470 → ~1300 GFLOPS |
| 合并 vs 跨步访存 | 332 vs 49 GB/s(6.8 倍差距) |
| warp shuffle 归约 | ~331 GB/s(读带宽 99% 利用) |
| 传输流水线(4 流) | 57 → 45 ms |
| fused 扩展 vs PyTorch 原生 | 1.26 → 2.53 ms |

数字会因机器而异,重要的是掌握"测量 → 定位 → 优化 → 复测"的闭环。

## 已知踩坑记录(本模块开发过程中实测)

- **PyTorch 扩展编译报 `Ninja is required`**:`pip install ninja` 后仍报错,
  是因为子进程 PATH 里没有环境 bin 目录。用
  `conda activate py312` 激活环境后运行(或 `PATH=$CONDA_PREFIX/bin:$PATH`)。
- **扩展算子校验 `torch.allclose` 失败**:GPU 一次乘加与 PyTorch 两步
  kernel 的浮点舍入顺序不同,默认容差(rt=1e-5, at=1e-8)对大数值不够,
  用 `rtol=1e-5, atol=1e-5` 显式容差。
- **cudaOccupancyMaxActiveBlocksPerMultiprocessor 显示 100% 但性能差**:
  占用率只是必要条件,还要看瓶颈分类(见阶段四 README 的表格)。

## 参考资料

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — 权威手册
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) — 优化方法论
- [Nsight Systems 文档](https://docs.nvidia.com/nsight-systems/) — 时间线分析
- [Nsight Compute 文档](https://docs.nvidia.com/nsight-compute/) — kernel 指标
- [CUDA Samples](https://github.com/NVIDIA/cuda-samples) — 官方示例库
- [CUDA Mode 讲座](https://www.youtube.com/@CUDAMODE) — 社区进阶课程

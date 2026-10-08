# 阶段六:PyTorch 自定义 CUDA 扩展(专家)

目标:把自己的 CUDA kernel 接入 PyTorch,完成"校验 → 接入当前流 →
正确性对照 → 性能对比"的完整工程闭环。这是把学到的 CUDA 知识
落到深度学习项目的最后一公里。

前置要求:完成阶段一~四;会基础 Python/PyTorch。

预计用时:3~5 小时。

## 环境

| 依赖 | 版本 | 说明 |
|------|------|------|
| conda env | `py312` | 本仓库标准环境 |
| PyTorch | 2.6.0+cu124 | 与系统 nvcc 12.4 版本一致,可直接编译 |
| CUDA Toolkit | 12.4 | `cpp_extension.load` 内部调用 nvcc |

运行方式(本阶段不进 Makefile,由 Python 驱动编译):

```bash
conda activate py312
cd 06_pytorch
python extension/demo.py          # 示例:fused saxpy
python solutions/train_fused.py   # 参考解答验证
```

## 核心概念

### 1. 数据接入:at::Tensor → 裸指针

```cpp
auto out = torch::empty_like(x);            // 分配输出,沿用 x 的设备/dtype
float *po = out.data_ptr<float>();          // 裸指针交给 kernel
```

`TORCH_CHECK` 校验设备、dtype、形状 —— 扩展代码的第一行防线。

### 2. 流对接:别用默认流

```cpp
kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(...);
```

PyTorch 自身是异步执行的:所有操作排在"当前流"上。自定义算子若用
默认流,轻则破坏与其它算子的顺序保证,重则数据竞争。这是自研算子
最容易踩的坑。

### 3. 融合 (fusion) 的价值

`a * x + y` 在 PyTorch 里是 2 次 kernel launch + 1 个中间张量的
显存读写;fused 版是 1 次 kernel、零中间张量。对短小算子链,
融合既是优化也是架构能力(FlashAttention 本质就是极致的融合)。

### 4. 两种编译方式

- **JIT(load)**:本模块用法,一行调用 nvcc,产物进 `extension/_build`,
  适合实验;
- **AOT(setup.py)**:`CUDAExtension` + `torch.utils.cpp_extension` 构建,
  适合正式发布。

## 文件说明

| 文件 | 说明 | 运行命令 |
|------|------|----------|
| `extension/saxpy_op.cu` | 扩展算子源码(saxpy + 原地 fused 版) | 被 demo.py 编译 |
| `extension/demo.py` | 编译扩展、正确性对照、CPU/原生/自定义三方计时 | `python extension/demo.py` |

## 练习

| 练习 | 说明 | 解答 |
|------|------|------|
| `exercises/fused_saxpy.cu` | 从零实现 `w1*x + w2*y + bias` 扩展 | `solutions/fused_saxpy.cu` + `solutions/train_fused.py` |

## 自检问题

- 为什么必须用 `getCurrentCUDAStream()`?用默认流可能出什么事故?
- `a.item<float>()` 会触发什么?在热路径上这样取标量好吗?
  (提示:隐式 D2H 同步,应提前在 Python 侧转成 float)
- 什么时候自定义 CUDA 算子比 PyTorch 组合原语更值得写?

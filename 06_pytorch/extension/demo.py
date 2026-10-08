"""demo.py — 编译并测试 PyTorch 自定义 CUDA 算子

学习点:
    1. torch.utils.cpp_extension.load:JIT 编译 extension(免 setup.py)
    2. 正确性验证:自定义算子 vs PyTorch 广播运算
    3. 性能对比:CPU / PyTorch 原生 / 自定义 CUDA 算子三方耗时

运行(需要 py312 环境):
    conda activate py312
    cd 06_pytorch && python extension/demo.py
"""
import os
import time
import torch
from torch.utils.cpp_extension import load

os.makedirs("extension/_build", exist_ok=True)

# JIT 编译:首次运行会调用 nvcc,约 1 分钟;之后走缓存
ext = load(
    name="saxpy_op",
    sources=["extension/saxpy_op.cu"],
    build_directory="extension/_build",   # 产物固定目录,便于 .gitignore
    verbose=True,
)


def check_correctness():
    n = 1 << 20
    a = torch.tensor(2.5, device="cuda")
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")

    got = ext.saxpy(a, x, y)
    want = a * x + y
    torch.cuda.synchronize()
    assert torch.allclose(got, want, rtol=1e-5, atol=1e-5), "saxpy 结果错误"
    print("正确性: 自定义算子与 PyTorch 原生结果一致 ✓")


def bench(fn, iters=100):
    # CUDA 计时:先预热,再同步后取平均
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000


def bench_speed():
    n = 1 << 24   # 1600 万元素
    a = torch.tensor(2.5, device="cuda")
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")

    t_cpu = bench(lambda: (2.5 * x.cpu() + y.cpu()))     # 反面教材:纯 CPU 计算
    t_torch = bench(lambda: (a * x + y))                 # 两次 kernel + 中间张量
    t_ext = bench(lambda: ext.saxpy(a, x, y))            # 一次 kernel
    t_fused = bench(lambda: ext.fused_saxpy(a, a, x, y)) # 一次 kernel,原地

    print(f"N={n}")
    print(f"  CPU 搬运版        : {t_cpu:8.3f} ms (含 H2D/D2H)")
    print(f"  PyTorch 原生      : {t_torch:8.3f} ms (2 kernel + 中间张量)")
    print(f"  自定义 fused 算子 : {t_ext:8.3f} ms (1 kernel)")
    print(f"  原地 fused 版     : {t_fused:8.3f} ms (1 kernel, 零分配)")


if __name__ == "__main__":
    print(f"PyTorch {torch.__version__}, CUDA 可用: {torch.cuda.is_available()}")
    check_correctness()
    bench_speed()

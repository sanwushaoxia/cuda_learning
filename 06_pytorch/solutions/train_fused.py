"""train_fused.py — 参考解答的测试脚本:编译、验证并计时 fused_affine

运行:
    conda activate py312
    cd 06_pytorch && python solutions/train_fused.py
"""
import os
import time
import torch
from torch.utils.cpp_extension import load

os.makedirs("solutions/_build", exist_ok=True)

ext = load(
    name="fused_affine_op",
    sources=["solutions/fused_saxpy.cu"],
    build_directory="solutions/_build",
    verbose=True,
)


def check_correctness():
    w1 = torch.tensor(1.5, device="cuda")
    w2 = torch.tensor(-0.5, device="cuda")
    bias = torch.tensor(0.25, device="cuda")
    x = torch.randn(1 << 20, device="cuda")
    y = torch.randn(1 << 20, device="cuda")

    got = ext.fused_affine(w1, w2, bias, x, y)
    want = w1 * x + w2 * y + bias
    torch.cuda.synchronize()
    assert torch.allclose(got, want, rtol=1e-5, atol=1e-5), "fused_affine 结果错误"
    print("正确性: fused_affine 与 PyTorch 原生结果一致 ✓")


def bench(fn, iters=100):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000


def bench_speed():
    n = 1 << 24
    w1 = torch.tensor(1.5, device="cuda")
    w2 = torch.tensor(-0.5, device="cuda")
    bias = torch.tensor(0.25, device="cuda")
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")

    t_torch = bench(lambda: (w1 * x + w2 * y + bias))   # 3 kernel + 2 中间张量
    t_ext = bench(lambda: ext.fused_affine(w1, w2, bias, x, y))

    print(f"N={n}")
    print(f"  PyTorch 原生(3 kernel): {t_torch:8.3f} ms")
    print(f"  fused 扩展(1 kernel)  : {t_ext:8.3f} ms")


if __name__ == "__main__":
    print(f"PyTorch {torch.__version__}, CUDA 可用: {torch.cuda.is_available()}")
    check_correctness()
    bench_speed()

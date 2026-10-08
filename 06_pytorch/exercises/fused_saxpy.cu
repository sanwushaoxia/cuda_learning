// fused_saxpy.cu — 练习:自己实现 fused saxpy PyTorch 扩展
//
// 任务:不参考 saxpy_op.cu,从零写一个 PyTorch 扩展算子:
//   out = w1 * x + w2 * y + bias
//
// 要求:
//   1. 写 fused_op.cu:kernel + C++ 包装函数(TORCH_CHECK 校验 +
//      at::cuda::getCurrentCUDAStream())
//   2. 写 train_fused.py:用 load() 编译,正确性对照
//      (w1*x + w2*y + bias 的 PyTorch 原生写法),并计时对比
//   3. 思考:和"两步 PyTorch 运算"相比,你省下了什么?
//      (答案:kernel launch 次数、中间张量的显存读写)
//
// 验收标准:
//   - torch.allclose 通过
//   - fused 版本耗时低于原生两步运算
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// TODO: 实现 kernel 与包装函数

// 参考答案思路(不看代码,只看结构):
//   kernel: out[i] = w1 * x[i] + w2 * y[i] + bias
//   包装:  TORCH_CHECK 设备/dtype/形状 -> empty_like ->
//          <<<grid, block, 0, getCurrentCUDAStream()>>> 启动 -> return out
//   python: ext = load(name=..., sources=[...]) -> 正确性 + 计时

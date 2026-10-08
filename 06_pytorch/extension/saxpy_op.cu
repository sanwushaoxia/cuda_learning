// saxpy_op.cu — PyTorch 自定义 CUDA 算子:fused saxpy
//
// 学习点:
//   1. 用 at::Tensor 接口访问数据:data_ptr<T>() 拿裸指针喂给 kernel
//   2. TORCH_CHECK 做输入校验(设备、dtype、形状)
//   3. 用 PyTorch 当前流(at::cuda::getCurrentCUDAStream())——
//      这样算子才能正确融入 PyTorch 的异步执行与流管理
//   4. 由 torch.utils.cpp_extension.load 编译,不需手动 nvcc
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void saxpy_kernel(float a, const float *x, const float *y, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a * x[i] + y[i];
}

torch::Tensor saxpy(torch::Tensor a, torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.is_cuda() && y.is_cuda(), "x/y 必须是 CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "仅支持 float32");
    TORCH_CHECK(x.sizes() == y.sizes(), "x/y 形状必须一致");

    auto out = torch::empty_like(x);
    const int n = x.numel();
    const float *px = x.data_ptr<float>();
    const float *py = y.data_ptr<float>();
    float *po = out.data_ptr<float>();
    const float a_val = a.item<float>();

    if (n == 0) return out;

    const int block = 256;
    const int grid = (n + block - 1) / block;
    // 关键:用 PyTorch 的当前流,而不是默认流
    saxpy_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        a_val, px, py, po, n);
    return out;
}

// fused 版:y = a*x + b*y 原地更新,演示"少一次读写"的融合价值
__global__ void fused_kernel(float a, float b, const float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + b * y[i];
}

torch::Tensor fused_saxpy(torch::Tensor a, torch::Tensor b,
                          torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.is_cuda() && y.is_cuda(), "x/y 必须是 CUDA tensor");
    auto out = torch::empty_like(x);
    const int n = x.numel();
    fused_kernel<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        a.item<float>(), b.item<float>(),
        x.data_ptr<float>(), y.data_ptr<float>(), n);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("saxpy", &saxpy, "fused saxpy: a*x + y");
    m.def("fused_saxpy", &fused_saxpy, "y = a*x + b*y");
}

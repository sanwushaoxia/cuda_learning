// fused_saxpy.cu — 练习参考解答:out = w1*x + w2*y + bias 的 PyTorch 扩展
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void fused_affine_kernel(float w1, float w2, float bias,
                                    const float *x, const float *y,
                                    float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = w1 * x[i] + w2 * y[i] + bias;
}

torch::Tensor fused_affine(torch::Tensor w1, torch::Tensor w2, torch::Tensor bias,
                           torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.is_cuda() && y.is_cuda(), "x/y must be CUDA tensors");
    TORCH_CHECK(x.scalar_type() == at::kFloat && y.scalar_type() == at::kFloat,
                "float32 only");
    TORCH_CHECK(x.sizes() == y.sizes(), "x/y shapes must match");

    auto out = torch::empty_like(x);
    const int n = x.numel();
    if (n == 0) return out;

    const int block = 256;
    const int grid = (n + block - 1) / block;
    fused_affine_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        w1.item<float>(), w2.item<float>(), bias.item<float>(),
        x.data_ptr<float>(), y.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_affine", &fused_affine, "out = w1*x + w2*y + bias");
}

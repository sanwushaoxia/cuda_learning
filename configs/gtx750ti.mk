# gtx750ti — Windows 机配置
# GPU: NVIDIA GeForce GTX 750 Ti (Maxwell, sm_50)
# Toolkit: CUDA 11.8(sm_50 支持的最后版本),经 tools/nvcc_win.cmd 编译
# 注意:CUDA 11.8 不兼容 VS 18,包装脚本固定用 VS 2019 BuildTools (MSVC 14.29)

GPU_ARCH        := sm_50
CUDA_HOME       := C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8

# 走 tools/nvcc_win.cmd(内部设 CUDA_PATH + VS2019 vcvars64)
USE_WIN_WRAPPER := 1

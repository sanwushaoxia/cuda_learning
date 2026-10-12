# rtx3060 — Linux 本机配置(默认)
# GPU: NVIDIA GeForce RTX 3060 (12GB, Ampere, sm_86)
# Toolkit: CUDA 12.4 (/usr/local/cuda)

GPU_ARCH        := sm_86
# 驱动支持 CUDA 13.0,但 toolkit 固定 12.4(与阶段六 PyTorch cu124 对齐)
CUDA_HOME       ?= /usr/local/cuda

# Linux 原生 nvcc,不走 Windows 包装脚本
USE_WIN_WRAPPER := 0

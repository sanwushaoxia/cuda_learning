# GPU 配置选择开关
#
# 可用 profile:
#   rtx3060   默认。Linux 本机,RTX 3060 (sm_86) + CUDA 12.4
#   gtx750ti  Windows 机,GTX 750 Ti (sm_50) + CUDA 11.8(经 tools/nvcc_win.cmd)
#
# 切换方式(二选一):
#   1. 改下面一行,长期生效
#   2. 命令行临时覆盖:make GPU_PROFILE=gtx750ti
#
# 各 profile 的产物在 build/<profile>/ 下隔离,切换后互不干扰。

GPU_PROFILE ?= rtx3060

include configs/$(GPU_PROFILE).mk

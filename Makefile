# CUDA 学习模块统一构建脚本
# 用法:
#   make all              编译全部示例(01-05 阶段)
#   make run T=vector_add 编译并运行指定示例
#   make profile T=vector_add   用 nsys 生成 profiling 报告
#   make clean            清理 build/ 目录
#
# 本机环境: RTX 3060 (sm_86) + CUDA 12.4

NVCC      ?= nvcc
NVCCFLAGS ?= -O3 -arch=sm_86 -std=c++17
BUILD_DIR := build
NSYS      ?= nsys

STAGES := 01_basics 02_memory 03_shmem 04_optimize 05_streams
SOURCES := $(foreach stage,$(STAGES),$(wildcard $(stage)/*.cu) $(wildcard $(stage)/exercises/*.cu) $(wildcard $(stage)/solutions/*.cu))
TARGETS := $(patsubst %.cu,$(BUILD_DIR)/%,$(SOURCES))

.PHONY: all run profile clean help
.DEFAULT_GOAL := help

all: $(TARGETS)

$(BUILD_DIR)/%: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $< -o $@

# 运行单个示例: make run T=vector_add (按文件名在各阶段查找)
run:
	@if [ -z "$(T)" ]; then echo "用法: make run T=<示例名>,如 make run T=vector_add"; exit 1; fi
	@exe=$$(find $(BUILD_DIR) -name '$(T)' -type f | head -1); \
	if [ -z "$$exe" ]; then echo "未找到示例 $(T),先 make all"; exit 1; fi; \
	echo "== 运行 $$exe =="; $$exe

# nsys profiling: make profile T=vector_add
profile:
	@if [ -z "$(T)" ]; then echo "用法: make profile T=<示例名>"; exit 1; fi
	@exe=$$(find $(BUILD_DIR) -name '$(T)' -type f | head -1); \
	if [ -z "$$exe" ]; then echo "未找到示例 $(T),先 make all"; exit 1; fi; \
	$(NSYS) profile -o $(BUILD_DIR)/$(T)_report --force-overwrite true $$exe && \
	$(NSYS) stats --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum $(BUILD_DIR)/$(T)_report.nsys-rep

clean:
	rm -rf $(BUILD_DIR)

help:
	@echo "CUDA 学习模块构建:"
	@echo "  make all                编译全部示例"
	@echo "  make run T=vector_add   编译并运行 (示例: T=hello_world / saxpy / matmul_tiled)"
	@echo "  make profile T=vector_add  nsys profiling 并输出 kernel/传输统计"
	@echo "  make clean              清理 build/"
	@echo ""
	@echo "可用示例:"
	@for t in $(notdir $(TARGETS)); do echo "  $$t"; done

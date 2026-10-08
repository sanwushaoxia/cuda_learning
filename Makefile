# CUDA 学习模块统一构建脚本
# 用法:
#   make all              编译全部示例(01-05 阶段)
#   make run T=vector_add 编译并运行指定示例
#   make profile T=vector_add   用 nsys 生成 profiling 报告
#   make clean            清理 build/ 目录
#
# 本机环境: GTX 750 Ti (sm_50) + CUDA 11.8

NVCC      ?= nvcc
NVCCFLAGS ?= -O3 -arch=sm_50 -std=c++17
BUILD_DIR := build
NSYS      ?= nsys
# Prefer CUDA 11.8 nvcc (supports sm_50). Windows build uses tools/nvcc_win.cmd.
CUDA_HOME ?= C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8

ifeq ($(OS),Windows_NT)
  EXE := .exe
  SHELL := C:/Progra~1/Git/bin/bash.exe
  .SHELLFLAGS := -c
  MKDIR_CMD = mkdir -p "$(1)"
  RM_CMD = rm -rf "$(BUILD_DIR)"
else
  EXE :=
  MKDIR_CMD = mkdir -p $(1)
  RM_CMD = rm -rf $(BUILD_DIR)
  NVCC := "$(CUDA_HOME)/bin/nvcc"
endif

STAGES := 01_basics 02_memory 03_shmem 04_optimize 05_streams
SOURCES := $(foreach stage,$(STAGES),$(wildcard $(stage)/*.cu) $(wildcard $(stage)/exercises/*.cu) $(wildcard $(stage)/solutions/*.cu))
TARGETS := $(patsubst %.cu,$(BUILD_DIR)/%$(EXE),$(SOURCES))

.PHONY: all run profile clean help
.DEFAULT_GOAL := help

all: $(TARGETS)

$(BUILD_DIR)/%$(EXE): %.cu
	@$(call MKDIR_CMD,$(dir $@))
ifeq ($(OS),Windows_NT)
	cmd.exe //C "tools\nvcc_win.cmd $(NVCCFLAGS) $< -o $@"
else
	$(NVCC) $(NVCCFLAGS) "$<" -o "$@"
endif

# Resolve source for T=name (searches stage / exercises / solutions)
SRC := $(firstword $(foreach stage,$(STAGES),$(wildcard $(stage)/$(T).cu) $(wildcard $(stage)/exercises/$(T).cu) $(wildcard $(stage)/solutions/$(T).cu)))
RUN_EXE := $(if $(SRC),$(BUILD_DIR)/$(basename $(SRC))$(EXE),)

# Run one example: make run T=vector_add (compile then execute)
run:
ifndef T
	$(error Usage: make run T=<name>, e.g. make run T=vector_add)
endif
ifeq ($(SRC),)
	$(error Example $(T) not found)
endif
	@$(MAKE) --no-print-directory $(RUN_EXE)
	$(info == Running $(RUN_EXE) ==)
	@$(RUN_EXE)

# nsys profiling: make profile T=vector_add
profile:
ifndef T
	$(error Usage: make profile T=<name>)
endif
ifeq ($(SRC),)
	$(error Example $(T) not found)
endif
	@$(MAKE) --no-print-directory $(RUN_EXE)
	$(NSYS) profile -o $(BUILD_DIR)/$(T)_report --force-overwrite true $(RUN_EXE) && \
	$(NSYS) stats --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum $(BUILD_DIR)/$(T)_report.nsys-rep

clean:
	@$(RM_CMD)

help:
	@echo "CUDA learning build:"
	@echo "  make all                  build all examples"
	@echo "  make run T=vector_add     build and run (e.g. T=hello_world / saxpy / matmul_tiled)"
	@echo "  make profile T=vector_add nsys profile + kernel/memcpy stats"
	@echo "  make clean                remove build/"
	@echo ""
	@echo "Available examples:"
	@for t in $(basename $(notdir $(TARGETS))); do echo "  $$t"; done

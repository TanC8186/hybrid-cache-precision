SHELL := /bin/bash
export ROOT := $(shell pwd)
PYTHON ?= python3

.PHONY: help setup-local setup-remote run sweep bench eval analyze archive reproduce artifact-check figures paper-dls check smoke-remote init-vendor lint test

help:  ## 显示所有命令
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-16s %s\n", $$1, $$2}'

init-vendor:  ## Reconstruct the pinned vLLM working tree
	./scripts/env/init_vllm.sh

setup-local:  ## 初始化本地开发环境（在 WSL2 内执行）
	./scripts/env/setup_wsl2.sh

setup-remote: ## 在租用的 5090 实例上校验并初始化运行环境
	./env_check.sh

run: ## 运行实验：make run EXP=<configs/experiments/<name>.yaml>
	./scripts/run.sh $(EXP)

sweep: ## 网格消融：make sweep CONFIG=<glob>
	python scripts/run.py --sweep --config $(if $(CONFIG),$(CONFIG),configs/quantization/*.yaml)

bench: ## 运行 serving benchmark：make bench MODE=memory|throughput|latency EXP=<name>
	python -m scripts.bench.$(MODE) --config configs/bench/$(MODE).yaml --experiment $(EXP)

eval: ## 运行质量评测：make eval TASK=perplexity|longbench|niah EXP=<name>
	python -m scripts.eval.$(TASK) --config configs/eval/quality.yaml --experiment $(EXP)

analyze: ## 汇总 metrics -> results/（唯一允许产出论文图表的入口）
	python -m scripts.analyze.aggregate

archive: ## 归档 headline 原始运行：make archive EXP=<name>
	./scripts/archive.sh $(EXP)

reproduce: ## 一键复现（供审稿人 / AE）
	bash ./reproduce.sh

artifact-check: ## CPU-only artifact integrity and unit-test gate
	PYTHON=$(PYTHON) bash ./reproduce.sh verify

figures: ## Regenerate publication figures from committed evidence
	PYTHON=$(PYTHON) bash ./reproduce.sh figures

paper-dls: ## Build the IEEE/DLS manuscript (requires latexmk)
	$(MAKE) -C paper/dls2026

check: ## 环境自检（本地或租机）
	./env_check.sh

smoke-remote: ## 远端 7B 冒烟：正式大规模运行前验证配置可跑
	./scripts/smoke_remote.sh

lint: ## 代码检查
	ruff check src scripts tests

test: ## 运行正确性门禁测试
	$(PYTHON) -m pytest -q tests/

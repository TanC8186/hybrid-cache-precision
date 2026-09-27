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

sweep: ## [not implemented] 网格消融：见 configs/experiments/ 与各 study 启动器
	@echo "make sweep is not implemented: scripts/run.py is a stub."
	@echo "Grid runs used the per-study launchers, e.g."
	@echo "  scripts/bench/run_capacity_phase_diagram.sh  (frozen contracts in configs/experiments/)"
	@exit 2

bench: ## [not implemented] serving benchmark：见 scripts/bench/ 的 5090 启动器
	@echo "make bench is not implemented: scripts.bench.{memory,throughput,latency} are stubs or absent."
	@echo "Serving runs used the 5090 launchers:"
	@echo "  scripts/bench/run_serving_formal.sh, scripts/bench/bench_driver_5090.sh, scripts/bench/bench_seed_loop.sh"
	@exit 2

eval: ## [not implemented] 质量评测：见 scripts/eval/run_*.sh 与对应 analyzer
	@echo "make eval is not implemented: scripts.eval.{perplexity,longbench,niah} are stubs."
	@echo "Quality stages ran through scripts/eval/run_*.sh (GSM8K, RULER, PPL) plus their"
	@echo "analyzers, e.g. scripts/eval/analyze_gsm8k_dependence.py; the PPL harness is"
	@echo "scripts/exp/hybrid_premise.py."
	@exit 2

analyze: ## [not implemented] 统一汇总入口：见各 stage analyzer 与 ./reproduce.sh figures
	@echo "make analyze is not implemented: scripts.analyze.aggregate is a stub."
	@echo "Paper values come from per-stage analyzers:"
	@echo "  scripts/bench/build_capacity_matrix_table.py"
	@echo "  scripts/eval/analyze_gsm8k_dependence.py"
	@echo "  scripts/quality/analyze_m4_formal.py"
	@echo "  paper/mlsys2026/figures/make_vector_figures.py  (via ./reproduce.sh figures)"
	@exit 2

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

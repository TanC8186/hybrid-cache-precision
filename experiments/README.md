# experiments — 运行原始产物（gitignored）

本目录**不入库**。每次运行在 `experiments/<name>/` 下产生：
- `resolved.yaml` + `.sha256`：解析后的有效配置
- `git_commit` / `vllm_submodule_status` / `git_dirty_stat`：代码状态
- `env_probe.txt`：环境探针
- `seeds.txt`：seed 清单
- `logs/`、`metrics.jsonl`、`checkpoints/`（校准产物）

**手工改动本目录视为无效**。headline 运行完成后 `make archive` 归档（zip+hash → results/_archive_index.txt）。
校准产物（scale/zero/clip）定义明确 schema，随 run 快照存储，作为 artifact 交付物。

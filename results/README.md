# results — 聚合分析结果（入库）

- `_provenance.jsonl`：run_id → config_hash / code_commit / vllm_sha / env_hash / data_hash / seeds（设计 §5.2）
- `ablations/*.csv`：网格消融输出（scripts/run.py --sweep）
- `tables/`：论文表格（csv / latex）
- `figures/`：论文图（pdf / svg）
- `_archive_index.txt`：headline 原始运行归档指针（scripts/archive.sh 写入）

**只由 `scripts/analyze` 产出**。任何手写/手改进 results 的内容视为无效。

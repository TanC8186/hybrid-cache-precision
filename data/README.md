# data — 数据集与 workload trace

- 大文件本体 **gitignored**（见根 .gitignore），溯源由 `data/MANIFEST.yaml` 承担
- 下载/更新走 `scripts/fetch_data.sh`（固定 revision）
- 生成数据（NIAH 等）走 `scripts/gen_synthetic_retrieval.py`
- 任何新数据文件都要登记进 `MANIFEST.yaml`（sha256 + 来源 + license + 预处理）

目录布局（下载后）：
```
datasets/     # LongBench、PG19、Wikitext 等评测数据集
niah/         # 合成检索数据（seed 化生成）
traces/       # serving 请求 trace（如 ShareGPT，注意重分发限制）
```

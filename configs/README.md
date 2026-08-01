# configs — 一切实验由 YAML 唯一定义

分层：
- `env/`            环境 profile（local_4060 / remote_5090），完整版本锁定
- `datasets/`       数据卡：url / split / sha256 / tokenizer / license / 预处理
- `models/`         模型定义（本地小模型 vs 远端 7B，同一家族）
- `quantization/`   量化方案（含 typed 校准块）
- `experiments/`    具体实验矩阵（强制 `seed` + 引擎默认值 + 测量协议）
- `bench/`          serving 扫描配置（throughput / latency）+ SLO
- `eval/`           质量评测配置 + 环境路由

约定：
- 每个实验一个 yaml，可引用 env/datasets/models/quantization 组合
- 运行入口 `scripts/run.sh` 固化**解析后**的配置快照（resolved.yaml + sha256）
- 用 schema 校验（见 `src/kvcache/utils` / tests），typo 快速失败

# bench_lat JSON Provenance（逐 rate_*.log Namespace 自动提取，2026-08-03）

> 补 code-review H3 缺口：E2/E3 的 20 个 JSON 自身不含 seed/warmup/server 配置；配置固化于各 rate_*.log 的 Namespace。本表由 `scripts/bench/gen_json_provenance.py` 生成。

| alloc | file | request_rate | num_prompts | seed | num_warmups | max_concurrency | random_input_len | random_output_len | base_url | dataset_name | model |
|---|---|---|---|---|---|---|---|---|---|---|---|
| int4 | rate_1.log | 1.0 | 400 | 0 | 0 | 512 | 1024 | 128 | http://127.0.0.1:8000 | random | /root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master |
| int4 | rate_12.log | 12.0 | 400 | 0 | 0 | 512 | 1024 | 128 | http://127.0.0.1:8000 | random | /root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master |
| int4 | rate_16.log | 16.0 | 400 | 0 | 0 | 512 | 1024 | 128 | http://127.0.0.1:8000 | random | /root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master |
| int4 | rate_20.log | 20.0 | 400 | 0 | 0 | 512 | 1024 | 128 | http://127.0.0.1:8000 | random | /root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master |
| int4 | rate_30.log | 30.0 | 400 | 0 | 0 | 512 | 1024 | 128 | http://127.0.0.1:8000 | random | /root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master |
| int4 | rate_4.log | 4.0 | 400 | 0 | 0 | 512 | 1024 | 128 | http://127.0.0.1:8000 | random | /root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master |
| int4 | rate_40.log | 40.0 | 400 | 0 | 0 | 512 | 1024 | 128 | http://127.0.0.1:8000 | random | /root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master |
| int4 | rate_50.log | 50.0 | 400 | 0 | 0 | 512 | 1024 | 128 | http://127.0.0.1:8000 | random | /root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master |
| int4 | rate_75.log | 75.0 | 400 | 0 | 0 | 512 | 1024 | 128 | http://127.0.0.1:8000 | random | /root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master |
| int4 | rate_8.log | 8.0 | 400 | 0 | 0 | 512 | 1024 | 128 | http://127.0.0.1:8000 | random | /root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master |
| fp16 | rate_1.log | 1.0 | 400 | 0 | 0 | 512 | 1024 | 128 | http://127.0.0.1:8001 | random | /root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master |
| fp16 | rate_12.log | 12.0 | 400 | 0 | 0 | 512 | 1024 | 128 | http://127.0.0.1:8001 | random | /root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master |
| fp16 | rate_16.log | 16.0 | 400 | 0 | 0 | 512 | 1024 | 128 | http://127.0.0.1:8001 | random | /root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master |
| fp16 | rate_20.log | 20.0 | 400 | 0 | 0 | 512 | 1024 | 128 | http://127.0.0.1:8001 | random | /root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master |
| fp16 | rate_30.log | 30.0 | 400 | 0 | 0 | 512 | 1024 | 128 | http://127.0.0.1:8001 | random | /root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master |
| fp16 | rate_4.log | 4.0 | 400 | 0 | 0 | 512 | 1024 | 128 | http://127.0.0.1:8001 | random | /root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master |
| fp16 | rate_40.log | 40.0 | 400 | 0 | 0 | 512 | 1024 | 128 | http://127.0.0.1:8001 | random | /root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master |
| fp16 | rate_50.log | 50.0 | 400 | 0 | 0 | 512 | 1024 | 128 | http://127.0.0.1:8001 | random | /root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master |
| fp16 | rate_75.log | 75.0 | 400 | 0 | 0 | 512 | 1024 | 128 | http://127.0.0.1:8001 | random | /root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master |
| fp16 | rate_8.log | 8.0 | 400 | 0 | 0 | 512 | 1024 | 128 | http://127.0.0.1:8001 | random | /root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master |

注释：`seed=0`, `num_warmups=0` 为 vllm bench serve 默认（本矩阵未显式传 --seed/--warmup-n）。server 层配置（kv_cache_dtype_per_layer 等）见 logs/PROVENANCE.md。

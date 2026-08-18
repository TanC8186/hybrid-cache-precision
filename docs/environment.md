# 环境说明

## 支持边界（2026-08-18 核实）

| 项 | 值 | 影响 |
|---|---|---|
| 宿主系统 | Windows 11 (WDDM) | **vLLM 无原生 Windows 支持** |
| 宿主 Python | 3.13.5 | **超出 vLLM/PyTorch 的 3.10-3.12 wheel 范围** |
| GPU | RTX 4060 Laptop 8GB, driver 581.42 | 本地 dev（sm_89 / Ada） |
| WSL2 | Ubuntu | 本地开发与 CPU artifact 验证环境 |
| Docker | 可选 | vLLM/CUDA 隔离环境 |

**结论**：本地开发必须走 WSL2（推荐）或 Docker。宿主 Windows 直跑 vLLM 不可行。

## 本地开发（4060，8GB，sm_89/Ada）

```bash
# 启动 WSL2 并进入项目
wsl -d Ubuntu -- bash -lc "cd /mnt/e/IEEE_MASS_Research && bash scripts/env/setup_wsl2.sh"
```

- Python 用 3.10-3.12（当前 `.venv` 为 3.12），不用宿主 3.13
- 移动或重命名项目目录后重建 `.venv`；激活脚本会记录创建时的绝对路径
- `/mnt/e`（NTFS）上构建 vLLM 较慢：可选在 WSL 的 ext4 文件系统中另建工作树
- 4060 结果**仅供 dev**，禁止进入 `results/`（内存/吞吐数字在 8G 卡上不可信）

## 租机（5090，32GB，sm_120/Blackwell）

**最终选型（2026-08-02 确认）**：
| 项 | 值 |
|---|---|
| CUDA | **13**（租机商唯一提供，匹配 torch cu130） |
| torch | **2.13.0+cu130**（pip 覆盖模板 torch，vLLM 硬性要求） |
| Python | **3.12** |
| 框架 | vLLM upstream `e2fa285` + 两个本地 patch commits，最终 `55f47685` |
| 驱动 | >= R580（Blackwell sm_120 必需） |
| 构建 | `TORCH_CUDA_ARCH_LIST="12.0"` |

**环境搭建**：先按 `vendor/README.md` 重建 vLLM，再执行
`bash scripts/env/setup_5090.sh` 与 `./env_check.sh`。冻结实验配置中的绝对路径
是原服务器 provenance；新机器应复制配置后适配，不能原地修改归档合同。

- 每次租用后先跑 `./env_check.sh`：确认 driver/CUDA 与锁定版本匹配、无残留进程、vLLM self-test
- **driver 跨租期漂移常见**：任何最终实验前必须校验
- 完整运行手册：`docs/notes/vllm-5090-runbook-2026-08-02.md`

## 关键版本矩阵

| 组件 | local_4060 | remote_5090 |
|---|---|---|
| 架构 | sm_89 (Ada) | sm_120 (Blackwell) |
| torch/vLLM wheel | 各架构独立编译，**二进制不互通** | 同左 |
| CUDA | 12.x（与 vLLM wheel 匹配） | 12.x（容器内） |

> vLLM/FlashAttention 按 CUDA 架构编译；同一 wheel 不能保证跨机器运行。
> 锁定源码见 `vendor/vllm-patches/`，运行时版本见各 verified artifact 的
> `environment.json`、contract 与哈希文件。

## 数值一致性

Ada 与 Blackwell 的舍入、累加顺序、TF32 行为不同 → **跨 GPU 数字不可混用**。
提交 `results/` 前必须跑跨 GPU 数值校验：quant-dequant roundtrip + 单 seed PPL，在两台机器上都跑一遍并核对。

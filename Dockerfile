# 权威运行环境：所有论文 headline 实验的容器化载体
# 基于固定的 vLLM 镜像 digest（与 configs/env/remote_5090.yaml 保持一致）
#
# TODO(实现集成后): 用 vendor/vllm 构建产出的 wheel 覆盖基础镜像，固定 digest
FROM vllm/vllm-openai:v0.8.4@sha256:TODO_PIN_DIGEST

# 进入容器后：安装本项目核心包 + 校验配置
WORKDIR /workspace
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

# 数据与配置通过 volume 挂载（data/ 与 configs/ 不入镜像）
# 运行：docker run --gpus all -v $PWD/configs:/workspace/configs -v $PWD/data:/workspace/data ...

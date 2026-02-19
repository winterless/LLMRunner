# Experiment: Qwen3-1.7B + Megatron

INCLUDE = "../../common/pipeline_megatron.py"

DATAPOOL_ROOT = "${DATAPOOL}/experiments/qwen3-1.7b_megatron_full"
DRY_RUN = 0

# Pipeline uses explicit Step Instance entries:
# - id: unique instance id (for logs/audit)
# - type: step type capability
# - config: optional per-instance config path (relative to config dir)
#
# This is the recommended form for modular, pluggable orchestration.
STEPS = [
    {"id": "tokenize_cpt_0", "type": "tokenize_cpt", "config": "steps/tokenize_cpt_0.py", "enabled": True},
    {"id": "tokenize_sft_0", "type": "tokenize_sft", "config": "steps/tokenize_sft_0.py", "enabled": True},
    {"id": "train_cpt_0", "type": "train_cpt", "config": "steps/train_cpt_0.py", "enabled": True},
    {"id": "mg2hf_0", "type": "mg2hf", "config": "steps/mg2hf_0.py", "enabled": True},
    {"id": "hf2mg_0", "type": "hf2mg", "config": "steps/hf2mg_0.py", "enabled": True},
    {"id": "train_sft_0", "type": "train_sft", "config": "steps/train_sft_0.py", "enabled": True},
]

# Base model configuration (single source of truth)
# BASE_MODEL_SRC: 原始模型路径，应直接指向包含 safetensors 的目录
# prepare_exp 会将 BASE_MODEL_SRC 复制到 ${DATAPOOL_ROOT}/model/base/${BASE_MODEL_NAME}
# 复制后，safetensors 文件直接在 ${DATAPOOL_ROOT}/model/base/${BASE_MODEL_NAME} 目录下
BASE_MODEL_NAME = "Qwen3-1.7B"
BASE_MODEL_SRC = "/home/unlimitediw/workspace/models/Qwen3-1.7B"
# BASE_MODEL_PATH: 实际模型在 datapool 中的路径（prepare_exp 后，供 steps 使用）
# 这个路径直接指向包含 safetensors、config.json、tokenizer.json 等的目录
# (defined in configs/common/pipeline_base.py)

# Model prefix for naming tokenized outputs/checkpoints
MODEL_PREFIX = "qwen3_1p7b"

# Data copy: set CPT_RAW_COPY_SRC and SFT_RAW_COPY_SRC in steps/tokenize_cpt_0.py and steps/tokenize_sft_0.py when needed.

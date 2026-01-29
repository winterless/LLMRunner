# Experiment: Qwen3-1.7B + NVIDIA (Megatron-DeepSpeed etc.).

# RUN_ID 仅用于日志子目录，产出路径按实验唯一（见 datapool/README）
RUN_ID = ""

DATAPOOL_ROOT = "datapool/experiments/qwen3-1.7b_nvidia_full"
WORKDIR = ".llmrunner"
DRY_RUN = 0

STEP_UDATASETS_ENABLED = 0
STEP_TOKENIZE_CPT_ENABLED = 1
STEP_TOKENIZE_SFT_ENABLED = 1
STEP_TRAIN_CPT_ENABLED = 1
STEP_TRAIN_SFT_ENABLED = 1
STEP_CONVERT_ENABLED = 1
STEP_EVAL_ENABLED = 0

# Base model configuration (single source of truth)
# BASE_MODEL_SRC: 原始模型路径，应直接指向包含 safetensors 的目录
# prepare_exp 会将 BASE_MODEL_SRC 复制到 ${DATAPOOL_ROOT}/model/base/${BASE_MODEL_NAME}
# 复制后，safetensors 文件直接在 ${DATAPOOL_ROOT}/model/base/${BASE_MODEL_NAME} 目录下
BASE_MODEL_NAME = "Qwen3-1.7B"
BASE_MODEL_SRC = "/home/unlimitediw/workspace/models/Qwen3-1.7B"
# BASE_MODEL_PATH: 实际模型在 datapool 中的路径（prepare_exp 后，供 steps 使用）
# 这个路径直接指向包含 safetensors、config.json、tokenizer.json 等的目录
BASE_MODEL_PATH = "${DATAPOOL_ROOT}/model/base/${BASE_MODEL_NAME}"

# Model prefix for naming tokenized outputs/checkpoints
MODEL_PREFIX = "qwen3_1p7b"

# Data copy: set CPT_RAW_COPY_SRC and SFT_RAW_COPY_SRC in steps/2.tokenize_cpt.py when needed.

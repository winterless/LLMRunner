# Experiment: Qwen3-4B + MindSpeed

INCLUDE = "../../common/pipeline_mindspeed.py"

DATAPOOL_ROOT = "${DATAPOOL}/experiments/qwen3-4b_mindspeed_full"
DRY_RUN = 1

STEP_UDATASETS_ENABLED = 0
STEP_TOKENIZE_CPT_ENABLED = 0
STEP_TOKENIZE_SFT_ENABLED = 0
STEP_TRAIN_CPT_ENABLED = 0
STEP_MG2HF_ENABLED = 0
STEP_HF2MG_ENABLED = 0
STEP_TRAIN_SFT_ENABLED = 0
STEP_CONVERT_ENABLED = 0
STEP_EVAL_ENABLED = 0

# Base model configuration (single source of truth)
# BASE_MODEL_SRC: 原始模型路径，应直接指向包含 safetensors 的目录
# prepare_exp 会将 BASE_MODEL_SRC 复制到 ${DATAPOOL_ROOT}/model/base/${BASE_MODEL_NAME}
# 复制后，safetensors 文件直接在 ${DATAPOOL_ROOT}/model/base/${BASE_MODEL_NAME} 目录下
BASE_MODEL_NAME = "Qwen3-4B"
BASE_MODEL_SRC = "/home/unlimitediw/workspace/models/Qwen3-4B"
# BASE_MODEL_PATH: 实际模型在 datapool 中的路径（prepare_exp 后，供 steps 使用）
# 这个路径直接指向包含 safetensors、config.json、tokenizer.json 等的目录
# (defined in configs/common/pipeline_base.py)

# Model prefix for naming tokenized outputs/checkpoints
MODEL_PREFIX = "qwen3_4b"


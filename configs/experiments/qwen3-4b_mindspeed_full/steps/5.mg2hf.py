# step5: MG -> HF size conversion (extern script)
EXTERN_SCRIPT = "echo 1"
MODEL_TYPE = "qwen3"

BASE_MODEL_PATH = "${BASE_MODEL_SRC}"
INPUT_MODEL_PATH = "${DATAPOOL_ROOT}/model/cpt_checkpoints"
OUTPUT_MODEL_PATH = "${DATAPOOL_ROOT}/model/hf"
STEPS = 1000
base_iter = 1
base_step = 1
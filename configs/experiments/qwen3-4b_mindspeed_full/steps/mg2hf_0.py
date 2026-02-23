# step5: MG -> HF size conversion (script)
SCRIPT = "echo 1"
MODEL_TYPE = "qwen3"

BASE_MODEL_PATH = "${BASE_MODEL_SRC}"
INPUT_MODEL_PATH = "${DATAPOOL_ROOT}/model/sft_checkpoints/mg_tp8/agent_neat_pack"
OUTPUT_MODEL_PATH = "${DATAPOOL_ROOT}/model/hf"
STEPS = 1312
base_iter = 1
base_step = 1

# Full export: MindSpeed checkpoint → HF (EXTERN_SCRIPT or CONVERT_CMD).
EXTERN_SCRIPT = "echo mg2hf_1_export"
INPUT_MODEL_PATH = "${DATAPOOL_ROOT}/model/sft_checkpoints/agent_neat_pack"
OUTPUT_MODEL_PATH = "${DATAPOOL_ROOT}/model/sft_checkpoints/hf"
BASE_MODEL_PATH = "${BASE_MODEL_SRC}"
MODEL_TYPE = "qwen3"

STEPS = 1000
base_iter = 1
base_step = 1

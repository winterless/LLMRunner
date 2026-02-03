# step6: HF -> MG size conversion (extern script)
EXTERN_SCRIPT = "echo 1"
MODEL_TYPE = "qwen3"

TOKENIZER_PATH = "${BASE_MODEL_SRC}"
INPUT_MODEL_PATH = "${DATAPOOL_ROOT}/model/hf"
OUTPUT_MODEL_PATH = "${DATAPOOL_ROOT}/model/cpt_checkpoints/tp8"
TP = 8
PP = 1
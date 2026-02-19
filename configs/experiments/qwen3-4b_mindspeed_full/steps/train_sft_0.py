# step7: MindSpeed SFT training (script)
# SCRIPT: 直接运行可替代本 step 的全部逻辑
SCRIPT = "echo 7.train sft"

# SFT train_config (exported as env for run_sft.sh)
INPUT_MODEL_PATH = "${DATAPOOL_ROOT}/model/hf/ckpt_1000b"
OUTPUT_MODEL_PATH = "${DATAPOOL_ROOT}/model/sft_checkpoints"
INPUT_DATA_PATH = "${DATAPOOL_ROOT}/data/tokenized/sft"
TOKENIZER_PATH = "${BASE_MODEL_PATH}"

PP = 1
CP = 1
TP = 8
GBS = 64
MBS = 1
SEQ_LEN = 32768
TRAIN_ITERS = 13120
SAVE_INTERVAL = 656
EVAL_INTERVAL = 1000
EVAL_ITERS = 0
SPLIT = "100,0,0"

LR = "1e-5"
MIN_LR = "1e-6"
WARMUP_FRACTION = "0.01"

ASCEND_LAUNCH_BLOCKING = 0
ASCEND_PROCESS_LOG = True
HCCL_CONNECT_TIMEOUT = 7000
TORCH_SHOW_CPP_STACKTRACES = 0
DEBUG_DUMP_ALIGN = 0

DATA_CACHE_PATH = "/"
EXP_NAME = "agent_neat_pack"
PROMPT_TYPE = "qwen"

# step7: MindSpeed SFT training (extern script)
# EXTERN_SCRIPT: 直接运行可替代本 step 的全部逻辑
EXTERN_SCRIPT = "${MINDSPEED}/scripts/run_sft.sh"

# SFT train_config (exported as env for run_sft.sh)
CKPT_LOAD_DIR = "/"
CKPT_SAVE_DIR = "/"
DATA_PATH = "/"
TOKENIZER_PATH = "/"

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

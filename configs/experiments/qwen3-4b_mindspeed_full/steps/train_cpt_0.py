# step4: MindSpeed CPT training (script)
# SCRIPT: 直接运行可替代本 step 的全部逻辑
SCRIPT = "${MINDSPEED}/scripts/run_cpt.sh"

# CPT train_config（原封不动，导出为环境变量供 run_cpt.sh 使用）
INPUT_MODEL_PATH = "${DATAPOOL_ROOT}/model/cpt_checkpoints"
OUTPUT_MODEL_PATH = "${DATAPOOL_ROOT}/model/cpt_checkpoints"
INPUT_DATA_PATH = "${DATAPOOL_ROOT}/data/tokenized/output"
TOKENIZER_PATH = "${BASE_MODEL_SRC}"
OUTPUT_PREFIX = ""
ds_prefix = "ds"

MODEL_TYPE = "qwen3_4b"
TP = 2
PP = 1
gbs = 1024

train_iters = 11922
save_interval = 3974

lr = "3e-4"
min_lr = "3e-5"
lr_decay_style = "cosine"
lr_decay_iters = 11922
lr_warmup_iters = 119

rotary_base = 10000
recompute_num_layers = 1

load_from_ckpt = False
FAULT_RECOVER = True
MA_RECOVER_POLICY = '{"actions":{"npu_proc_restart":{"max_num":3,"downgrade":"job_reschedule_with_taint"},"proc_restart":{"max_num":3},"job_reschedule":{"max_num":3},"job_reschedule_with_taint":{"max_num":3}},"exception_handle":[]}'

ASCEND_PROCESS_LOG = True
hccl_port = 64000
master_port = 13284

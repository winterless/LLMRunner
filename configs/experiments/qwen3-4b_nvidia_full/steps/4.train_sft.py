# step4: NVIDIA SFT. Megatron 做 SFT 一般用 pretrain_gpt.py 加载 CPT checkpoint + SFT 数据（或项目内 finetune 脚本），
# 通过 TRAIN_CMD 写完整命令。不设 ENTRYPOINT（posttrain_gpt.py 是 MindSpeed 专用）。
RUN_WITH = "cmd"
TRAINER_DIR = "/path/to/Megatron-DeepSpeed"
SFT_RAW_GLOB = "${DATAPOOL_ROOT}/data/raw/sft/*.jsonl"
LOAD_DIR = "${DATAPOOL_ROOT}/model/cpt_checkpoints"
SAVE_DIR = "${DATAPOOL_ROOT}/model/sft_checkpoints"
# Example: same as CPT but --load from LOAD_DIR, data-path to SFT tokenized data, and SFT hyperparams.
TRAIN_CMD = ""

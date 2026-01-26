# step3: NVIDIA CPT (e.g. Megatron-DeepSpeed pretrain_gpt.py)
# RUN_WITH=cmd → run TRAIN_CMD (set below). RUN_WITH=entrypoint → run python ENTRYPOINT ARGS. No default.
RUN_WITH = "cmd"
TRAINER_DIR = "/path/to/Megatron-DeepSpeed"
DATA_PATH = "${DATAPOOL_ROOT}/data/tokenized/cpt/qwen3_4b_text_document"
SAVE_DIR = "${DATAPOOL_ROOT}/model/cpt_checkpoints"
# Full command (example single-GPU). Edit for your run.
TRAIN_CMD = """torchrun --nproc_per_node 1 pretrain_gpt.py \\
  --data-path ${DATA_PATH} \\
  --save ${SAVE_DIR} \\
  --vocab-file ${DATAPOOL_ROOT}/model/base/Qwen3-4B/tokenizer.json \\
  --merge-file '' \\
  --log-interval 10"""
ENTRYPOINT = "pretrain_gpt.py"
ARGS = ""

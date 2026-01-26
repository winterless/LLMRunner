# step5: NVIDIA checkpoint → HF. RUN_WITH=cmd → run CONVERT_CMD (set in this file). RUN_WITH=entrypoint → python ENTRYPOINT ARGS.
RUN_WITH = "cmd"
TRAINER_DIR = "/path/to/Megatron-DeepSpeed"
IN_CKPT_DIR = "${DATAPOOL_ROOT}/model/sft_checkpoints"
OUT_HF_DIR = "${DATAPOOL_ROOT}/model/hf"
CONVERT_CMD = ""
ENTRYPOINT = "tools/convert_checkpoint.py"
ARGS = ""

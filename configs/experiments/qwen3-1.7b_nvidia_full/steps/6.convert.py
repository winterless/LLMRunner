# step5: NVIDIA checkpoint → HF. RUN_WITH=cmd → run CONVERT_CMD (set in this file). RUN_WITH=entrypoint → python ENTRYPOINT ARGS.
RUN_WITH = "cmd"
TRAINER_DIR = "/home/unlimitediw/workspace/Megatron-LM"
MEGATRON_DIR = "/home/unlimitediw/workspace/Megatron-LM"
CONDA_ENV = "LLMTrain"
IN_CKPT_DIR = "${DATAPOOL_ROOT}/model/sft_checkpoints"
OUT_HF_DIR = "${DATAPOOL_ROOT}/model/hf"
# Converter: Megatron-LM tools/checkpoint (Megatron -> HF, no TE)
CONVERT_CMD = """PYTHONPATH=${ROOT_DIR}/scripts:${MEGATRON_DIR}/tools/checkpoint:${MEGATRON_DIR}:$PYTHONPATH python ${MEGATRON_DIR}/tools/checkpoint/convert.py \
  --model-type GPT \
  --loader legacy_nf \
  --saver hf_qwen3 \
  --load-dir ${IN_CKPT_DIR} \
  --save-dir ${OUT_HF_DIR} \
  --megatron-path ${MEGATRON_DIR} \
  --loader-transformer-impl local"""
# Copy tokenizer/config before conversion (HF output expects these)
COPY_HF_BEFORE = 1
COPY_HF_BEFORE_SUBDIR = ""
COPY_HF_FROM = "${BASE_MODEL_PATH}"
COPY_HF_FILES = "config.json,tokenizer.json,tokenizer_config.json,special_tokens_map.json,generation_config.json"
COPY_HF_OVERWRITE = 0
ENTRYPOINT = ""
ARGS = ""

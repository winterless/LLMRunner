# step5: NVIDIA checkpoint → HF. RUN_WITH=cmd → run CONVERT_CMD (set in this file). RUN_WITH=entrypoint → python ENTRYPOINT ARGS.
RUN_WITH = "cmd"
TRAINER_DIR = "/home/unlimitediw/workspace/MindSpeed"
MEGATRON_DIR = "/home/unlimitediw/workspace/Megatron-LM"
CONDA_ENV = "LLMTrain"
IN_CKPT_DIR = "${DATAPOOL_ROOT}/model/sft_checkpoints"
OUT_HF_DIR = "${DATAPOOL_ROOT}/model/hf"
# Converter: MindSpeed convert_ckpt.py (Megatron -> HF)
CONVERT_CMD = """PYTHONPATH=${TRAINER_DIR}:$PYTHONPATH python convert_ckpt.py \
  --model-type GPT \
  --loader mg \
  --saver "" \
  --load-dir ${IN_CKPT_DIR} \
  --save-dir ${OUT_HF_DIR} \
  --megatron-path ${MEGATRON_DIR} \
  --model-type-hf qwen3 \
  --save-model-type hf \
  --load-from-legacy"""
# Copy config/tokenizer before conversion (required by MindSpeed HF loader)
COPY_HF_BEFORE = 1
COPY_HF_BEFORE_SUBDIR = ""
COPY_HF_FROM = "${BASE_MODEL_PATH}"
COPY_HF_FILES = "config.json,tokenizer.json,tokenizer_config.json,special_tokens_map.json,generation_config.json"
COPY_HF_OVERWRITE = 0
# Copy tokenizer/config into final HF dir (OUT_HF_DIR/mg2hf)
COPY_HF_SUBDIR = "mg2hf"
ENTRYPOINT = "convert_ckpt.py"
ARGS = ""

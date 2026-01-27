# step4: NVIDIA CPT (e.g. Megatron-DeepSpeed pretrain_gpt.py)
# RUN_WITH=cmd → run TRAIN_CMD (set below). RUN_WITH=entrypoint → run python ENTRYPOINT ARGS. No default.
RUN_WITH = "cmd"
TRAINER_DIR = "/home/unlimitediw/workspace/Megatron-LM"
DATA_PATH = "${DATAPOOL_ROOT}/data/tokenized/cpt/qwen3_4b_text_document"
SAVE_DIR = "${DATAPOOL_ROOT}/model/cpt_checkpoints"
# BASE_MODEL_PATH is defined in pipeline.py (single source of truth)
# Use ${BASE_MODEL_PATH} which will be resolved from pipeline config
BASE_MODEL_PATH = "${BASE_MODEL_PATH}"

# Qwen3-4B model parameters (from config.json)
# hidden_size: 2560, num_layers: 36, num_attention_heads: 32, vocab_size: 151936
# num_key_value_heads: 8 (GQA), head_dim: 128, intermediate_size: 9728
# max_position_embeddings: 32768

# Full training command for single GPU (RTX 5090, 32GB)
# Note: For quick validation, we use very few steps (20 steps, ~2-3 minutes)
# To load from base model, first convert HF to Megatron format, then use --load
TRAIN_CMD = """conda run -n LLMTrain torchrun --nproc_per_node=1 --master_port=29500 pretrain_gpt.py \\
  --data-path ${ROOT_DIR}/${DATA_PATH} \\
  --save ${ROOT_DIR}/${SAVE_DIR} \\
  --tokenizer-model ${BASE_MODEL_PATH} \\
  --tokenizer-type HuggingFaceTokenizer \\
  --merge-file '' \\
  --num-layers 36 \\
  --hidden-size 2560 \\
  --num-attention-heads 32 \\
  --ffn-hidden-size 9728 \\
  --seq-length 256 \\
  --max-position-embeddings 32768 \\
  --vocab-size 151936 \\
  --micro-batch-size 1 \\
  --global-batch-size 1 \\
  --train-iters 20 \\
  --recompute-granularity selective \\
  --lr 6.0e-4 \\
  --min-lr 6.0e-5 \\
  --lr-decay-style cosine \\
  --lr-warmup-iters 2 \\
  --weight-decay 0.1 \\
  --adam-beta1 0.9 \\
  --adam-beta2 0.95 \\
  --adam-eps 1.0e-8 \\
  --optimizer-cpu-offload \\
  --clip-grad 1.0 \\
  --log-interval 5 \\
  --save-interval 10 \\
  --eval-interval 1000 \\
  --eval-iters 10 \\
  --tensorboard-dir ${ROOT_DIR}/${SAVE_DIR}/tensorboard \\
  --bf16 \\
  --no-masked-softmax-fusion \\
  --no-bias-gelu-fusion \\
  --transformer-impl local \\
  --disable-bias-linear \\
  --normalization RMSNorm \\
  --no-persist-layer-norm \\
  --no-gradient-accumulation-fusion \\
  --swiglu \\
  --untie-embeddings-and-output-weights"""
ENTRYPOINT = "pretrain_gpt.py"
ARGS = ""

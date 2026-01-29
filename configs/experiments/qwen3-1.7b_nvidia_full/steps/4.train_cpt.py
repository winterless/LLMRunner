# step4: NVIDIA CPT (e.g. Megatron-DeepSpeed pretrain_gpt.py)
# RUN_WITH=cmd → run TRAIN_CMD (set below). RUN_WITH=entrypoint → run python ENTRYPOINT ARGS. No default.
RUN_WITH = "cmd"
TRAINER_DIR = "/home/unlimitediw/workspace/Megatron-LM"
DATA_PATH = "${DATAPOOL_ROOT}/data/tokenized/cpt/${MODEL_PREFIX}_text_document"
SAVE_DIR = "${DATAPOOL_ROOT}/model/cpt_checkpoints"
# BASE_MODEL_PATH is defined in pipeline.py (single source of truth)
# Use ${BASE_MODEL_PATH} which will be resolved from pipeline config
BASE_MODEL_PATH = "${BASE_MODEL_PATH}"

# Qwen3-1.7B model parameters (from config.json)
# hidden_size: 2048, num_layers: 28, num_attention_heads: 16, vocab_size: 151936
# intermediate_size: 6144, max_position_embeddings: 40960

# Full training command for single GPU (RTX 5090, 32GB)
# 显存优化策略（按优先级）:
#   1. CPU Offload 三件套（必须一起用）: 节省优化器状态显存约 6-7GB
#   2. Flash Attention: 减少激活值显存
#   3. Recompute selective: 用计算换显存
#   4. BF16: 减少参数/激活显存
#   5. 降低 seq-length: 如果还是 OOM，可以降到 256 或更小
# 显存配置: seq=512, batch=1 + CPU offload 约需 12-14GB（不含checkpoint）
# 注意: save-interval=1000 避免在2步训练时保存checkpoint
# no-save-optim: 不保存优化器状态，减少checkpoint保存时的显存占用（但无法resume训练）
# checkpoint格式: 使用legacy格式（--ckpt-format torch），保存为单个.pt文件
# For quick validation, we use very few steps (2 steps)
TRAIN_CMD = """torchrun --nproc_per_node=1 --master_port=29500 pretrain_gpt.py \\
  --data-path ${ROOT_DIR}/${DATA_PATH} \\
  --save ${ROOT_DIR}/${SAVE_DIR} \\
  --tokenizer-model ${BASE_MODEL_PATH} \\
  --tokenizer-type HuggingFaceTokenizer \\
  --merge-file '' \\
  --num-layers 4 \\
  --hidden-size 512 \\
  --num-attention-heads 8 \\
  --ffn-hidden-size 2048 \\
  --seq-length 256 \\
  --max-position-embeddings 40960 \\
  --position-embedding-type rope \\
  --rotary-base 1000000 \\
  --vocab-size 151936 \\
  --micro-batch-size 1 \\
  --global-batch-size 1 \\
  --train-iters 2 \\
  --recompute-granularity selective \\
  --lr 6.0e-4 \\
  --min-lr 6.0e-5 \\
  --lr-decay-style cosine \\
  --lr-warmup-iters 1 \\
  --weight-decay 0.1 \\
  --adam-beta1 0.9 \\
  --adam-beta2 0.95 \\
  --adam-eps 1.0e-8 \\
  --clip-grad 1.0 \\
  --log-interval 5 \\
  --save-interval 1000 \\
  --eval-interval 1000 \\
  --eval-iters 10 \\
  --reset-attention-mask \\
  --reset-position-ids \\
  --tensorboard-dir ${ROOT_DIR}/${SAVE_DIR}/tensorboard \\
  --ckpt-format torch \\
  --no-save-optim \\
  --bf16 \\
  --use-flash-attn \\
  --attention-backend flash \\
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

  # --use-distributed-optimizer \\
  # --use-precision-aware-optimizer \\
  #   --optimizer-cpu-offload \\
  # --optimizer-offload-fraction 0.5 \\
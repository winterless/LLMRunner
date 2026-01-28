# step4: NVIDIA SFT (Megatron-LM pretrain_gpt.py finetune mode)
#
# 目标：从 CPT checkpoint 加载权重（--load），用 tokenize_sft 产物训练/验证最小闭环。
# 注意：Megatron-LM 默认 --ckpt-format=torch_dist（会产出 .distcp）。
# 这里显式指定 legacy：--ckpt-format torch（产出单个 .pt）。
RUN_WITH = "cmd"
TRAINER_DIR = "/home/unlimitediw/workspace/Megatron-LM"

# Input / output
DATA_PATH = "${DATAPOOL_ROOT}/data/tokenized/sft/qwen3_4b_sft_packed_text_document"
LOAD_DIR = "${DATAPOOL_ROOT}/model/cpt_checkpoints"
SAVE_DIR = "${DATAPOOL_ROOT}/model/sft_checkpoints"

# Tokenizer: single source of truth from pipeline.py
BASE_MODEL_PATH = "${BASE_MODEL_PATH}"

# Minimal “run-through” config (single GPU)
TRAIN_CMD = """conda run -n LLMTrain torchrun --nproc_per_node=1 --master_port=29501 pretrain_gpt.py \\
  --data-path ${ROOT_DIR}/${DATA_PATH} \\
  --load ${ROOT_DIR}/${LOAD_DIR} \\
  --save ${ROOT_DIR}/${SAVE_DIR} \\
  --ckpt-format torch \\
  --finetune \\
  --tokenizer-model ${BASE_MODEL_PATH} \\
  --tokenizer-type HuggingFaceTokenizer \\
  --merge-file '' \\
  --num-layers 28 \\
  --hidden-size 2048 \\
  --num-attention-heads 16 \\
  --ffn-hidden-size 6144 \\
  --seq-length 512 \\
  --max-position-embeddings 40960 \\
  --vocab-size 151936 \\
  --micro-batch-size 1 \\
  --global-batch-size 1 \\
  --train-iters 2 \\
  --split 100,0,0 \\
  --recompute-granularity selective \\
  --lr 6.0e-5 \\
  --min-lr 6.0e-6 \\
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
  --eval-iters 0 \\
  --reset-attention-mask \\
  --reset-position-ids \\
  --tensorboard-dir ${ROOT_DIR}/${SAVE_DIR}/tensorboard \\
  --no-save-optim \\
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

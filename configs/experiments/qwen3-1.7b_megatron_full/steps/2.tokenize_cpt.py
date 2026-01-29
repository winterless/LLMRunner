# step2: Megatron-LM preprocess_data.py → CPT tokenize → tokenized/cpt/ (bin/idx)
# Raw copy: CPT 源在下方，prepare_exp 会拷贝到 data/raw/cpt
CPT_RAW_COPY_SRC = "/home/unlimitediw/workspace/TYDeepResearch/UDatasets/out/mixed/mymix/tmp"
INPUT_DIR = "${DATAPOOL_ROOT}/data/raw/cpt"
# TOKENIZER_MODEL: Use BASE_MODEL_PATH from pipeline.py (single source of truth)
TOKENIZER_MODEL = "${BASE_MODEL_PATH}"
# HuggingFaceTokenizer 通常不需要显式指定 vocab-file，tokenizer 会自动从 TOKENIZER_MODEL 路径找到 tokenizer.json
# 如果遇到 tokenizer 加载问题，可以取消注释下面这行：
# TOKENIZER_VOCAB_FILE = "${DATAPOOL_ROOT}/model/base/Qwen3-4B/Qwen3-4B-Base/tokenizer.json"
OUTPUT_PREFIX = "${DATAPOOL_ROOT}/data/tokenized/cpt/${MODEL_PREFIX}"
MEGATRON = "${MEGATRON}"
CONDA_ENV = "LLMTrain"
WORKERS = 32
# PARTITIONS>1 时，Megatron 会为每个 key 生成 .idx（第 404 行在循环内 finalize）
PARTITIONS = 1
LOG_INTERVAL = 100000
JSON_KEYS = "text"
TOKENIZER_TYPE = "HuggingFaceTokenizer"
# 注意：INPUT_DIR 可以是目录路径或单个文件路径（不支持 glob 模式）
# 目录中的所有 .jsonl 文件会自动合并为一个文件再处理（排除分区文件如 *_0.jsonl）

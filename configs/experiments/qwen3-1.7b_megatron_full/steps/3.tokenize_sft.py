# step2b: Megatron-LM preprocess_data.py → SFT tokenize → tokenized/sft/ (bin/idx)
# Raw copy: SFT 源在下方，prepare_exp 会拷贝到 data/raw/sft
SFT_RAW_COPY_SRC = "/home/unlimitediw/workspace/TYDeepResearch/UDatasets/out/mixed/mymix/sft"
INPUT_DATA_PATH = "${DATAPOOL_ROOT}/data/raw/sft"
OUTPUT_PREFIX = "${DATAPOOL_ROOT}/data/tokenized/sft/${MODEL_PREFIX}_sft_packed"
# TOKENIZER_PATH: Use BASE_MODEL_PATH from pipeline.py (single source of truth)
TOKENIZER_PATH = "${BASE_MODEL_PATH}"
# HuggingFaceTokenizer 通常不需要显式指定 vocab-file，tokenizer 会自动从 TOKENIZER_PATH 路径找到 tokenizer.json
# 如果遇到 tokenizer 加载问题，可以尝试取消注释下面这行：
# TOKENIZER_VOCAB_FILE = "${BASE_MODEL_PATH}/tokenizer.json"
# SFT jsonl 字段常与 CPT 不同：无 "text" 时需设 JSON_KEYS（空格分隔，与 preprocess_data --json-keys 一致）
# 将原始数据统一转换为 input/label/text，再用 text 做 tokenizer
REWRITE_INPUT_LABEL = 1
REWRITE_OUTPUT_FILE = "${DATAPOOL_ROOT}/data/raw/sft/sft_input_label.jsonl"
PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n"
PROMPT_INPUT_TEMPLATE = "### Input:\n{input}\n"
PROMPT_RESPONSE_PREFIX = "### Response:\n"
JSON_KEYS = "text"
MEGATRON = "${MEGATRON}"
WORKERS = 32
# PARTITIONS>1 时，Megatron 会为每个 key 生成 .idx（第 404 行在循环内 finalize）
PARTITIONS = 1
LOG_INTERVAL = 100000
TOKENIZER_TYPE = "HuggingFaceTokenizer"
# 注意：INPUT_DATA_PATH 可以是目录路径或单个文件路径（不支持 glob 模式）
# 目录中的所有 .jsonl 文件会自动合并为一个文件再处理（排除分区文件如 *_0.jsonl）

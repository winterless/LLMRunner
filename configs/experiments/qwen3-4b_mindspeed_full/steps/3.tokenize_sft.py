# step3: MindSpeed preprocess_data.py → SFT tokenize → tokenized/sft/ (bin/idx)
# Raw copy: SFT 源在下方，prepare_exp 会拷贝到 data/raw/sft
SFT_RAW_COPY_SRC = "/home/unlimitediw/workspace/TYDeepResearch/UDatasets/out/mixed/mymix/sft"
INPUT_DIR = "${DATAPOOL_ROOT}/data/raw/sft"
OUTPUT_PREFIX = "${DATAPOOL_ROOT}/data/tokenized/sft/${MODEL_PREFIX}_sft_packed"
# TOKENIZER_MODEL: Use BASE_MODEL_PATH from pipeline.py (single source of truth)
TOKENIZER_MODEL = "${BASE_MODEL_PATH}"

# 将原始数据统一转换为 input/label/text，再用 text 做 tokenizer
REWRITE_INPUT_LABEL = 1
REWRITE_OUTPUT_FILE = "${DATAPOOL_ROOT}/data/raw/sft/sft_input_label.jsonl"
PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n"
PROMPT_INPUT_TEMPLATE = "### Input:\n{input}\n"
PROMPT_RESPONSE_PREFIX = "### Response:\n"
JSON_KEYS = "text"

# MindSpeed directory
MINDSPEED = "${MINDSPEED}"

WORKERS = 32
# PARTITIONS>1 时，preprocess 会为每个 key 生成 .idx
PARTITIONS = 1
LOG_INTERVAL = 100000
TOKENIZER_TYPE = "HuggingFaceTokenizer"


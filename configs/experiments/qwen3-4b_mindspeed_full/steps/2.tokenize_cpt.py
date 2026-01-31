# step2: MindSpeed preprocess_data.py → CPT tokenize → tokenized/cpt/ (bin/idx)
# Raw copy: CPT 源在下方，prepare_exp 会拷贝到 data/raw/cpt
CPT_RAW_COPY_SRC = "/home/unlimitediw/workspace/TYDeepResearch/UDatasets/out/mixed/mymix/tmp"
INPUT_DIR = "${DATAPOOL_ROOT}/data/raw/cpt"
# TOKENIZER_MODEL: Use BASE_MODEL_PATH from pipeline.py (single source of truth)
TOKENIZER_MODEL = "${BASE_MODEL_PATH}"
OUTPUT_PREFIX = "${DATAPOOL_ROOT}/data/tokenized/cpt/${MODEL_PREFIX}"

# MindSpeed directory
MINDSPEED = "${MINDSPEED}"

WORKERS = 32
# PARTITIONS>1 时，preprocess 会为每个 key 生成 .idx
PARTITIONS = 1
LOG_INTERVAL = 100000
JSON_KEYS = "text"
TOKENIZER_TYPE = "HuggingFaceTokenizer"
MERGE_JSONL=1
SHUFFLE_JSONL=1

# step2: Megatron-LM preprocess_data.py → CPT tokenize → tokenized/cpt/ (bin/idx)
# Raw copy: CPT 源在下方，prepare_exp 会拷贝到 data/raw/cpt
SCRIPT = "python3 ${ROOT_DIR}/scripts/steps/tokenize_cpt.py"
CPT_RAW_COPY_SRC = "/home/unlimitediw/workspace/TYDeepResearch/UDatasets/out_for_LLRunner/cpt_data"
INPUT_DATA_PATH = "${DATAPOOL_ROOT}/data/raw/cpt"
OUTPUT_PREFIX = "${DATAPOOL_ROOT}/data/tokenized/cpt/${MODEL_PREFIX}"
MEGATRON = "${MEGATRON}"
WORKERS = 32
PARTITIONS = 1
LOG_INTERVAL = 100000
JSON_KEYS = "text"
TOKENIZER_TYPE = "HuggingFaceTokenizer"

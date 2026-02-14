# step2b: Megatron-LM preprocess_data.py → SFT tokenize → tokenized/sft/ (bin/idx)
# Raw copy: SFT 源在下方，prepare_exp 会拷贝到 data/raw/sft
SFT_RAW_COPY_SRC = "/home/unlimitediw/workspace/TYDeepResearch/UDatasets/out_for_LLRunner/sft_data"
INPUT_DATA_PATH = "${DATAPOOL_ROOT}/data/raw/sft"
OUTPUT_PREFIX = "${DATAPOOL_ROOT}/data/tokenized/sft/${MODEL_PREFIX}_sft_packed"
REWRITE_INPUT_LABEL = 1
REWRITE_OUTPUT_FILE = "${DATAPOOL_ROOT}/data/raw/sft/sft_input_label.jsonl"
PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n"
PROMPT_INPUT_TEMPLATE = "### Input:\n{input}\n"
PROMPT_RESPONSE_PREFIX = "### Response:\n"
JSON_KEYS = "text"
MEGATRON = "${MEGATRON}"
WORKERS = 32
PARTITIONS = 1
LOG_INTERVAL = 100000
TOKENIZER_TYPE = "HuggingFaceTokenizer"

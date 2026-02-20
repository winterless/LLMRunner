# step2: MindSpeed preprocess_data.py → sft tokenize → tokenized/sft/ (bin/idx)
SFT_RAW_COPY_SRC = "/home/unlimitediw/workspace/TYDeepResearch/UDatasets/out_for_LLRunner/sft_data"
INPUT_DATA_PATH = "${DATAPOOL_ROOT}/data/raw/sft"

MERGE_JSONL = 1
SHUFFLE_JSONL = 1

INPUT_DATA_FILE = "${DATAPOOL_ROOT}/data/raw/sft/merged_input.jsonl"
OUTPUT_DATA_PATH = "${DATAPOOL_ROOT}/data/tokenized/sft"
TOKENIZER_PATH = "${BASE_MODEL_PATH}"
SCRIPT = "echo 2.tokenizer"

# step2: MindSpeed preprocess_data.py -> CPT tokenize -> tokenized/cpt/ (bin/idx)
CPT_RAW_COPY_SRC = "/home/unlimitediw/workspace/TYDeepResearch/UDatasets/out_for_LLRunner/cpt_data"
INPUT_DATA_PATH = "${DATAPOOL_ROOT}/data/raw/cpt"

MERGE_JSONL = 1
SHUFFLE_JSONL = 1

INPUT_DATA_FILE = "${DATAPOOL_ROOT}/data/raw/cpt/merged_input.jsonl"
OUTPUT_DATA_PATH = "${DATAPOOL_ROOT}/data/tokenized/cpt"
TOKENIZER_PATH = "${BASE_MODEL_PATH}"
SCRIPT = "echo 2.tokenizer"

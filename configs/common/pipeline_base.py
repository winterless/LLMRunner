# Shared pipeline defaults (invariant across experiments)

# RUN_ID only affects log subdir; outputs stay per-experiment.
RUN_ID = ""
WORKDIR = ".llmrunner"

# Base model path (derived from DATAPOOL_ROOT + BASE_MODEL_NAME)
BASE_MODEL_PATH = "${DATAPOOL_ROOT}/model/base/${BASE_MODEL_NAME}"

# Default tokenizer paths (override per experiment if needed)
TOKENIZER_PATH = "${BASE_MODEL_PATH}"
SFT_TOKENIZER_PATH = "${BASE_MODEL_PATH}"

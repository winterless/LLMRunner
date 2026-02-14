# Shared pipeline defaults (invariant across experiments)
#
# Pipeline steps: set STEPS to a list of step names (order and repeats allowed).
# Step types: udatasets, tokenize_cpt, tokenize_sft, train_cpt, mg2hf, hf2mg, train_sft, eval.
# Example: STEPS = ["tokenize_cpt", "tokenize_sft", "train_cpt", "mg2hf", "hf2mg", "train_sft"]
# mg2hf/hf2mg are atomic; mg2hf can also do full export (CONVERT_CMD+copy) via config.
# If STEPS is not set, legacy STEP_*_ENABLED=1 is used with default order.

# RUN_ID only affects log subdir; outputs stay per-experiment.
RUN_ID = ""
WORKDIR = ".llmrunner"

# Base model path (derived from DATAPOOL_ROOT + BASE_MODEL_NAME)
BASE_MODEL_PATH = "${DATAPOOL_ROOT}/model/base/${BASE_MODEL_NAME}"

# Default tokenizer paths (override per experiment if needed)
TOKENIZER_PATH = "${BASE_MODEL_PATH}"
SFT_TOKENIZER_PATH = "${BASE_MODEL_PATH}"

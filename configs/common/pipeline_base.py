# Shared pipeline defaults (invariant across experiments)
#
# Pipeline steps: set STEPS to a list of step instances (recommended) or step names.
# Step types: tokenize_cpt, tokenize_sft, train_cpt, mg2hf, hf2mg, train_sft, eval.
# Example (recommended):
# STEPS = [
#   {"id": "cpt_tok", "type": "tokenize_cpt"},
#   {"id": "cpt_stage1", "type": "train_cpt", "config": "steps/cpt_stage1.py"},
# ]
# mg2hf/hf2mg are atomic; mg2hf can also do full export (CONVERT_CMD+copy) via config.
# STEPS is required and must be defined explicitly in pipeline.py.

# RUN_ID only affects log subdir; outputs stay per-experiment.
RUN_ID = ""
WORKDIR = ".llmrunner"

# Base model path (derived from DATAPOOL_ROOT + BASE_MODEL_NAME)
BASE_MODEL_PATH = "${DATAPOOL_ROOT}/model/base/${BASE_MODEL_NAME}"

# Default tokenizer paths (override per experiment if needed)
TOKENIZER_PATH = "${BASE_MODEL_PATH}"
SFT_TOKENIZER_PATH = "${BASE_MODEL_PATH}"

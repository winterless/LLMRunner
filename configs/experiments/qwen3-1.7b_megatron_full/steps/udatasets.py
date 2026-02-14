# step1: UDatasets - Optional step to prepare JSONL data
# This step only calls a shell command you configure.
# If you already prepared jsonl manually, disable this step.
UDATASETS_OUTPUT_DIR = "${DATAPOOL_ROOT}/data/processed"
UDATASETS_OUTPUT_URI = "obs://datasets/<name>/<ver>/processed/"
UDATASETS_CMD = ""

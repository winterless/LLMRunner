# step2b: Megatron-LM preprocess_data.py → SFT tokenize → tokenized/sft/ (bin/idx)
# Raw copy: SFT 源在下方，prepare_exp 会拷贝到 data/raw/sft
SFT_RAW_COPY_SRC = "/home/unlimitediw/workspace/TYDeepResearch/UDatasets/out/mixed/mymix/sft"
INPUT_DIR = "${DATAPOOL_ROOT}/data/raw/sft"
OUTPUT_PREFIX = "${DATAPOOL_ROOT}/data/tokenized/sft/qwen3_4b_sft"
TOKENIZER_MODEL = "${DATAPOOL_ROOT}/model/base/Qwen3-4B/Qwen3-4B-Base"
# HuggingFaceTokenizer 通常不需要显式指定 vocab-file，tokenizer 会自动从 TOKENIZER_MODEL 路径找到 tokenizer.json
# 如果遇到 tokenizer 加载问题，可以尝试取消注释下面这行：
# TOKENIZER_VOCAB_FILE = "${DATAPOOL_ROOT}/model/base/Qwen3-4B/Qwen3-4B-Base/tokenizer.json"
# SFT jsonl 字段常与 CPT 不同：无 "text" 时需设 JSON_KEYS（空格分隔，与 preprocess_data --json-keys 一致）
# 例如 Alpaca 等: instruction input output ；若为单字段可设 JSON_KEYS="content" 等
# 注意：PARTITIONS>1 时 Megatron 会为每个 key 生成 .idx；PARTITIONS=1 时只生成最后一个 key 的 .idx（Megatron bug）
JSON_KEYS = "instruction input output"
MEGATRON_DIR = "/home/unlimitediw/workspace/Megatron-LM"
WORKERS = 32
# PARTITIONS>1 时，Megatron 会为每个 key 生成 .idx（第 404 行在循环内 finalize）
PARTITIONS = 1
LOG_INTERVAL = 100000
TOKENIZER_TYPE = "HuggingFaceTokenizer"
# 注意：INPUT_DIR 可以是目录路径或单个文件路径（不支持 glob 模式）
# 目录中的所有 .jsonl 文件会自动合并为一个文件再处理（排除分区文件如 *_0.jsonl）

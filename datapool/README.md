## datapool

本目录是 **LLMRunner 的唯一数据入口/出口**，避免 pipeline 去仓库外的目录直接读写数据。

### 按实验唯一

每个**实验**（一套 config，如 `configs/experiments/qwen3-4b_nvidia_full/`）对应一套 datapool 产出路径，**不再用 RUN_ID 分层**。同一实验多次执行 pipeline 时，产出写入同一组目录（会覆盖）。需要对照实验时，请新建一个 experiment（复制配置与数据），见 README 中的 `prepare_exp --clone-experiment`。

### 目录约定

- `data/raw/`：原始数据（jsonl）
  - `raw/cpt/`：CPT 用 jsonl
  - `raw/sft/`：SFT 用 jsonl
- `data/processed/`：UDatasets 处理后的数据（jsonl）
- `data/tokenized/`：tokenize 输出，**CPT 与 SFT 分开**，各自 bin/idx：
  - `tokenized/cpt/`：CPT 的 `.bin`、`.idx`（step3 train_cpt 读）
  - `tokenized/sft/`：SFT 的 `.bin`、`.idx`（step4 train_sft 读）
- `model/cpt_checkpoints/`：CPT checkpoint（mg）
- `model/sft_checkpoints/`：SFT checkpoint（mg）
- `model/hf/`：convert 输出（HF safetensors 等）
- `model/base/`：base 模型（prepare_exp 拷贝）
- `reports/`：eval 输出（报告/指标）


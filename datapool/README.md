## datapool

本目录是 **LLMRunner 的唯一数据入口/出口**，避免 pipeline 去仓库外的目录直接读写数据。

建议约定：

- `datapool/data/raw/`：原始数据（jsonl 等）
- `datapool/data/processed/`：UDatasets 处理后的数据（jsonl）
- `datapool/data/tokenized/`：tokenize 输出（bin/idx + manifest）
- `datapool/model/cpt_checkpoints/`：CPT checkpoint（mg）
- `datapool/model/sft_checkpoints/`：SFT checkpoint（mg）
- `datapool/model/hf/`：convert 输出（HF safetensors 等）
- `datapool/reports/`：eval 输出（报告/指标）
- `datapool/intermediates/<run_id>/`：每次 pipeline 运行的“隔离工作区”（中间/临时产物区），避免不同 run 的产物互相覆盖


# Architecture

本项目当前的总体架构以 `LLMRunner1.drawio` 为单一事实来源（流程、边界、产物口径以图为准）。

## Diagram

- `LLMRunner1.drawio`: 端到端流程（数据→制品→训练→转换→归档→评测→报告）+ 实验过程记录

## Dataflow (from `LLMRunner.drawio`)

### Inputs → Data prep

- **原始数据**：多来源/多 schema（包含 CPT/SFT 候选样本）
- **Udatasets 处理**：数据增强、schema 校验、去重/清洗
  - **输出**：CPT/SFT 数据（manifest）

### Unified Registry (Artifact Store)

- **制品仓库**：OBS + Manifest + 版本号  
  - 示例：`obs://datasets/{name}/{ver}/`
  - 统一承载：
    - Udatasets 产出的 CPT/SFT 数据 manifest
    - Tokenize 产出的 `bin/idx + manifest`

### Jobs (training + packaging)

- **Tokenize 作业**
  - 输入：制品仓库中的数据 manifest
  - 输出：`bin/idx + manifest` 回写制品仓库
- **CPT 训练**
  - 输入：制品仓库中的 tokenized dataset（manifest）
  - 输出：`mg checkpoint + metadata`（作为 SFT 的基座）
- **SFT 训练**
  - 输入：制品仓库中的 SFT 数据（manifest）+ **CPT 基座（mg checkpoint）**
  - 输出：`mg checkpoint + metadata`
- **统一转换/打包（唯一一处）**
  - 输入：SFT 的 `mg checkpoint`
  - 输出：HF(safetensors) + tokenizer/config/README（供下游消费）

### Eval / Release

- **归档模型，测试准备**：将本次实验/模型产物进入可评测状态
- **评测 Runner**
  - 输入：归档模型/评测集
  - 输出：评测报告/指标
- **评测报告/指标**
  - 示例：`obs://reports/{model}/{run_id}/`

### Experiment tracking

- **实验过程记录**：对训练/评测的关键参数、运行信息、结果指标等做统一记录（与作业解耦）

## Key design principles

- **单一制品入口**：CPT/SFT 数据统一由 Udatasets 产出并落制品仓库，避免多条数据线并行演进。
- **基座显式化**：SFT 的 base 明确来自 CPT checkpoint，减少“线绕来绕去”的隐式依赖。
- **转换单点**：统一转换/打包保持唯一实现，降低重复脚本与格式漂移。

## Repo structure (minimal, shell-first)

```
LLMRunner/
  LLMRunner.drawio
  architecture.md
  README.md

  configs/
    pipeline.env               # enable/disable steps, DRY_RUN, workdir
    steps/
      udatasets.env
      tokenize.env
      train_cpt.env
      train_sft.env
      convert.env
      eval.env

  scripts/
    run.sh                     # shell runner (logs, run_id, orchestration)
    steps/
      udatasets.sh
      tokenize.sh
      train_cpt.sh
      train_sft.sh
      convert.sh
      eval.sh
```

### Execution model

- `scripts/run.sh` 只负责：
- 读取 `configs/pipeline.env`（现由 `python3 scripts/run.py` 管理；`scripts/run.sh` 为 wrapper）
  - 决定每步是否启用（`STEP_<STEP>_ENABLED=1`）
  - 统一日志目录：`${WORKDIR}/logs/${RUN_ID}/<step>.log`
  - `DRY_RUN=1` 时只打印命令
- 每个 step 脚本只负责“如何跑”：
  - 读取自己的 `configs/steps/<step>.env`
  - 组装并执行你指定的命令（Megatron/MindSpeed/bfcl_v3/任意 shell）


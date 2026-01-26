# LLMRunner

最小化的 LLM 训练流水线运行管理器。

## 快速开始

```bash
# 1. 准备实验环境（创建 datapool 目录，拷贝 base 模型和原始数据）
python3 scripts/prepare_exp.py -c configs/experiments/<实验名>/pipeline.env

# 2. 运行流水线
python3 scripts/run.py -c configs/experiments/<实验名>/pipeline.env
```

### 创建新实验

```bash
# 仅复制配置
python3 scripts/prepare_exp.py --clone-experiment qwen3-4b_nvidia_full my_new_exp

# 复制配置并复制 datapool（数据、模型、报告）
python3 scripts/prepare_exp.py --clone-experiment qwen3-4b_nvidia_full my_new_exp --copy-datapool
```

## 流水线步骤

1. **udatasets**: 数据增强、schema 校验、去重清洗
2. **tokenize_cpt**: CPT 数据 tokenization（生成 `.bin`/`.idx`）
3. **tokenize_sft**: SFT 数据 tokenization（生成 `.bin`/`.idx`）
4. **train_cpt**: CPT 预训练
5. **train_sft**: SFT 微调
6. **convert**: 模型转换（Megatron → HuggingFace）
7. **eval**: 模型评测

通过 `pipeline.env` 中的 `STEP_*_ENABLED=1/0` 控制各步骤的启用。

## 配置系统

- **流水线配置**: `configs/experiments/<实验名>/pipeline.env` - 控制步骤启用/禁用
- **步骤配置**: `configs/experiments/<实验名>/steps/<N>.<step>.py` - Python 配置文件

配置支持变量替换（如 `${DATAPOOL_ROOT}`），详见 `scripts/utils/config.py`。

## 数据目录结构

产出路径按**实验唯一**（无 RUN_ID 分层），详见 `datapool/README.md`：

```
datapool/experiments/<实验名>/
├── data/
│   ├── raw/          # 原始数据（jsonl）
│   │   ├── cpt/      # CPT 数据
│   │   └── sft/      # SFT 数据
│   ├── processed/    # UDatasets 处理后数据
│   └── tokenized/    # Tokenize 输出
│       ├── cpt/      # CPT .bin/.idx
│       └── sft/      # SFT .bin/.idx
├── model/
│   ├── base/              # Base 模型（prepare_exp 拷贝）
│   ├── cpt_checkpoints/  # CPT checkpoint
│   ├── sft_checkpoints/  # SFT checkpoint
│   └── hf/               # Convert 输出（HF 格式）
└── reports/              # Eval 输出
```

## 特性

- ✅ **自动清空输出目录**: 每次运行前自动清空步骤输出目录，避免旧文件残留
- ✅ **Python 优先**: 所有步骤脚本和配置均为 Python，更易维护
- ✅ **实验隔离**: 每个实验独立的 datapool，便于对照实验
- ✅ **灵活配置**: 支持变量替换、条件启用步骤

## 文档

- **架构设计**: `architecture.md`
- **流程图**: `LLMRunner1.drawio`
- **数据目录说明**: `datapool/README.md`

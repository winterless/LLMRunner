# LLMRunner

最小化的 LLM 训练流水线运行管理器。

## 快速开始

```bash
# 1. 准备实验环境（创建 datapool 目录，拷贝 base 模型和原始数据）
python3 scripts/prepare_exp.py -c configs/experiments/<实验名>/pipeline.py

# 2. 运行流水线
python3 scripts/run.py -c configs/experiments/<实验名>/pipeline.py
```

或者使用便捷脚本：

```bash
./run.sh [config_path]
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
5. **mg2hf**: MG→HF（原子：EXTERN_SCRIPT；或完整导出：CONVERT_CMD + COPY_HF_*）
6. **hf2mg**: HF→MG（原子，EXTERN_SCRIPT）
7. **train_sft**: SFT 微调
8. **eval**: 模型评测

通过 `pipeline.py` 中的 `STEP_*_ENABLED = 1/0` 控制各步骤的启用。

当前流水线提供 tokenize、train_cpt、train_sft、转换等标准形式，后续计划支持可插拔式步骤编排。

## 配置系统

- **流水线配置**: `configs/experiments/<实验名>/pipeline.py` - Python 配置文件，控制步骤启用/禁用和模型路径
- **步骤配置**: `configs/experiments/<实验名>/steps/<N>.<step>.py` - Python 配置文件

### 模型路径配置

模型路径配置统一在 `pipeline.py` 中管理（单一数据源）：

```python
# BASE_MODEL_SRC: 原始模型路径，应直接指向包含 safetensors 的目录
BASE_MODEL_SRC = "/path/to/model/directory"

# BASE_MODEL_NAME: 模型名称（用于 datapool 中的目录名）
BASE_MODEL_NAME = "Qwen3-1.7B"

# BASE_MODEL_PATH: 实际模型在 datapool 中的路径（自动派生）
# prepare_exp 会将 BASE_MODEL_SRC 复制到 ${DATAPOOL_ROOT}/model/base/${BASE_MODEL_NAME}
BASE_MODEL_PATH = "${DATAPOOL_ROOT}/model/base/${BASE_MODEL_NAME}"
```

配置支持变量替换（如 `${DATAPOOL_ROOT}`, `${BASE_MODEL_PATH}`），详见 `scripts/utils/config.py`。

## 数据目录结构

产出路径按**实验唯一**（无 RUN_ID 分层），详见 `datapool/README.md`：

```
datapool/experiments/<实验名>/
├── data/
│   ├── raw/          # 原始数据（jsonl）
│   │   ├── cpt/      # CPT 数据
│   │   └── sft/      # SFT 数据
│   ├── processed/    # UDatasets 处理后数据（step1 udatasets 输出）
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
- ✅ **统一模型路径**: 模型路径配置集中在 `pipeline.py`，单一数据源

## 文档

- **架构设计**: `architecture.md`
- **流程图**: `LLMRunner1.drawio`
- **数据目录说明**: `datapool/README.md`


## 世界知识

- **1. 大模型搜寻世界知识** 
    gemini, gpt等强大模型搜索世界数据，从huggingface，github等来源获取数据集
- **2. 数据集切片分析**
    数据集如agent data collection的随机采样信息，喂给AI Coder (Cursor, Trae等)，分析出数据特征，设计出对于的数据adapter（详见UDataset adapter模块）；这一步也会拿一些论文的先验指导
- **3. UDataset数据处理**
    数据集喂给UDataset清洗，然后Adapter进行数据重构
- **4. 模型数据训练与评测**
    数据喂给LLMRunner完成端到端训练与评测，记录评测结果
- **5. 结果反馈**
    增益数据归档，增益与负向数据分别总结&抽样喂给（1）搜索大模型-优化搜索过程 （2）AI Coder-优化adapter。并泛化宗教界
- **备注** 
    流程完全自动化，可并行；低原始数据要求（无格式要求）；增量评测集提升数据泛化读
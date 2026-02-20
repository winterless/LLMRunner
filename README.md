# LLMRunner

LLMRunner 是一个以 `pipeline.py` 为中心的最小训练编排器，负责：

- 准备实验目录和数据（`prepare_exp.py`）
- 按 `STEPS` 执行 step 实例（`run.py`）
- 统一把 step 配置解析为环境变量并执行 `SCRIPT`

## 简要设计思路

- **显式实例化编排**：`pipeline.py` 必须定义 `STEPS`，每个实例用 `id/type/config/enabled` 描述。
- **单一 prepare 归口**：raw 数据 copy/merge 只在 `scripts/prepare_exp.py` 处理。
- **实例命名约束**：step config 采用 `<step_type>_<idx>.py`（例如 `tokenize_cpt_0.py`）。

## 代码框架介绍

### 核心入口

- `scripts/run.py`
  - 读取 `pipeline.py`
  - 解析并校验 `STEPS`
  - 调用 `prepare_exp.prepare_from_env(...)`
  - 逐个执行启用的 step 实例
- `scripts/prepare_exp.py`
  - 创建 datapool 目录
  - 拷贝 base model
  - 扫描 `steps/tokenize_cpt_<idx>.py`、`steps/tokenize_sft_<idx>.py`
  - 执行 raw copy 与 merge/shuffle

### Step 脚本

- 路径：`scripts/steps/*.py`
- step 类型：`tokenize_cpt`, `tokenize_sft`, `train_cpt`, `mg2hf`, `hf2mg`, `train_sft`, `eval`
- 每个实例通过环境变量接收配置（如 `STEP_ENV_PATH`, `DATAPOOL_ROOT`, `MODEL_PREFIX` 等）

### 配置结构

- 实验配置：`configs/experiments/<experiment>/pipeline.py`
- step 配置：`configs/experiments/<experiment>/steps/<step_type>_<idx>.py`

## 最简执行命令

```bash
# 1) 预处理（准备 datapool、base 模型、raw copy/merge）
python3 scripts/prepare_exp.py -c configs/experiments/<experiment>/pipeline.py

# 2) 运行 pipeline（内部仍会调用 prepare）
python3 scripts/run.py -c configs/experiments/<experiment>/pipeline.py
```

可选：

```bash
# 只做 prepare，不执行 steps
python3 scripts/run.py -c configs/experiments/<experiment>/pipeline.py --prepare-only
```

## 参数介绍

### `scripts/run.py`

- `-c, --config`：必填，`pipeline.py` 路径
- `--prepare-only`：只执行 prepare，跳过 step 执行

### `scripts/prepare_exp.py`

- `-c, --config`：`pipeline.py` 路径（默认模式）

## `STEPS` 约束（与实现一致）

`pipeline.py` 示例：

```python
STEPS = [
    {"id": "tokenize_cpt_0", "type": "tokenize_cpt", "config": "steps/tokenize_cpt_0.py", "enabled": True},
    {"id": "train_cpt_0", "type": "train_cpt", "config": "steps/train_cpt_0.py", "enabled": True},
]
```

约束：

- `id` 必须是 `type_idx`（例如 `train_cpt_0`, `train_cpt_1`）
- 若设置 `config`，文件名 stem 必须等于 `id`
- `enabled=False` 仅影响 step 执行，不影响 prepare

## 数据目录约定

每个实验独立写入：

```text
datapool/experiments/<experiment>/
  data/raw/{cpt,sft}
  data/tokenized/{cpt,sft}
  model/{base,cpt_checkpoints,sft_checkpoints,hf}
  reports
```
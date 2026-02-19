# 实验配置说明

## steps/ 下的配置文件

每个 step 类型对应**一个**配置文件，文件名即类型名：`<step_name>.py`。

- **脚本**：`scripts/steps/<step_name>.py`（如 `tokenize_cpt.py`）
- **配置**：`configs/experiments/<实验>/steps/<step_name>.py`（如 `tokenize_cpt.py`）

**Step 类型**：udatasets, tokenize_cpt, tokenize_sft, train_cpt, mg2hf, hf2mg, train_sft, eval。  
mg2hf / hf2mg 为原子步骤；mg2hf 也可做完整导出（CONVERT_CMD + COPY_HF_*），可用 mg2hf_0.py / mg2hf_1.py 区分。

推荐在 step 配置中统一使用 `SCRIPT` 作为执行入口（可指向工程内 python 脚本或外部 shell 命令）。

**执行顺序与重复**：由 pipeline 的 `STEPS` 列表决定，推荐使用显式 Step Instance：

```python
STEPS = [
  {"id": "tokenize_cpt_0", "type": "tokenize_cpt", "config": "steps/tokenize_cpt_0.py"},
  {"id": "train_cpt_0", "type": "train_cpt", "config": "steps/cpt_stage1.py"},
  {"id": "train_cpt_1", "type": "train_cpt", "config": "steps/cpt_stage2.py"},
]
```

兼容旧写法（字符串列表）：

```python
STEPS = ["tokenize_cpt", "train_cpt", "train_cpt"]
```

当显式指定 `config` 时，推荐每个实例使用与 `id` 一致的配置文件名（如 `id=train_cpt_0` 对应 `steps/train_cpt_0.py`）。

`id` 规范必须是 `type_idx`（如 `train_cpt_0`、`train_cpt_1`），与是否设置 `config` 无关；不符合会报错。  
若设置 `config`，其文件名（stem）必须与 `id` 一致（如 `id=train_cpt_0` -> `steps/train_cpt_0.py`）。
脚本内可通过环境变量区分实例：`STEP_ID`、`STEP_TYPE`、`STEP_INDEX`、`STEP_OCCURRENCE_INDEX`。

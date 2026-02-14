# 实验配置说明

## steps/ 下的配置文件

每个 step 类型对应**一个**配置文件，文件名即类型名：`<step_name>.py`。

- **脚本**：`scripts/steps/<step_name>.py`（如 `tokenize_cpt.py`）
- **配置**：`configs/experiments/<实验>/steps/<step_name>.py`（如 `tokenize_cpt.py`）

**Step 类型**：udatasets, tokenize_cpt, tokenize_sft, train_cpt, mg2hf, hf2mg, train_sft, eval。  
mg2hf / hf2mg 为原子步骤；mg2hf 也可做完整导出（CONVERT_CMD + COPY_HF_*），可用 mg2hf_0.py / mg2hf_1.py 区分。

**执行顺序与重复**：由 pipeline 的 `STEPS` 列表决定，例如：

```python
STEPS = ["tokenize_cpt", "tokenize_sft", "train_cpt", "train_cpt", "train_sft", "eval"]
```

同一类型出现多次时，可为每次使用单独配置：

- 第一次（index=0）：优先用 `steps/train_cpt_0.py`，不存在则用 `steps/train_cpt.py`
- 第二次（index=1）：优先用 `steps/train_cpt_1.py`，不存在则用 `steps/train_cpt.py`

例如两次 CPT 用不同参数时，在实验下增加 `train_cpt_0.py` 和 `train_cpt_1.py` 即可。脚本内可通过环境变量 `STEP_INDEX` 区分第几次执行。

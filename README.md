## LLMRunner

最小化的 **shell 流水线运行管理器**：

- 每个流程一个 shell：`scripts/steps/<step>.sh`
- 每个 shell 一个配置文件：`configs/steps/<step>.env`
- 一个总入口 runner：`python3 scripts/run.py -c configs/pipeline.env`（`scripts/run.sh` 只是 wrapper）

### Quickstart

```bash
# 先 dry-run（只打印命令，不执行）
python3 scripts/run.py -c configs/pipeline.env

# 真跑：把 configs/pipeline.env 里的 DRY_RUN=0
```

### Single source of truth

- 流程图：`LLMRunner1.drawio`
- 架构文档：`architecture.md`


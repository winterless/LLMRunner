# Experiment Archive Template (Full Params)

Use this template to archive each experiment run. The goal is to capture all
step-level parameters, inputs/outputs, logs, and results for later review.

## Summary
- experiment: <exp_name>
- run_id: <run_id>
- status: <success|failed>
- started_at: <YYYY-MM-DD HH:MM:SS>
- finished_at: <YYYY-MM-DD HH:MM:SS>
- duration_sec: <int>
- git_commit: <sha>
- git_dirty: <true|false>
- host: <hostname>
- user: <user>
- os: <uname>
- python: <version>
- conda_env: <env>
- cuda: <version>
- cuda_driver: <version>
- torch: <version>
- gpus: <model x count>

## Dependencies (Versions)
| name | version |
| --- | --- |
| transformers | <version> |
| megatron | <version_or_commit> |
| apex | <version_or_not_installed> |
| transformer_engine | <version_or_not_installed> |
| vllm | <version_or_n/a> |

## Config Snapshot
- pipeline_config: <path>
- steps_config_dir: <path>
- pipeline_resolved:
  <KEY>: <VALUE>
- steps_resolved:
  <step_config_path>:
    <KEY>: <VALUE>
    ...

## Paths
- datapool_root: <path>
- workdir: <path>
- logs_root: <path>

## Data Snapshot
- raw_cpt_dir: <path>
- raw_sft_dir: <path>
- merged_cpt: <path> (lines=<int>, bytes=<int>)
- merged_sft: <path> (lines=<int>, bytes=<int>)

## Data Copy (Prepare)
| source | destination | files_copied | files_skipped | note |
| --- | --- | --- | --- | --- |
| <CPT_RAW_COPY_SRC> | <datapool/raw/cpt> | <int> | <int> | <optional> |
| <SFT_RAW_COPY_SRC> | <datapool/raw/sft> | <int> | <int> | <optional> |

## Steps
- step_id: 1
  name: <tokenize|train|mg2hf|hf2mg|convert|eval|udatasets|...>
  stage: <cpt|sft|other|none>
  status: <success|failed|skipped>
  config: <step_config_path>
  inputs:
    <key>: <value>
  outputs:
    <key>: <value>
  params:
    # Full parameters (resolved step config + runtime args dump)
    <KEY>: <VALUE>
  training:
    # Only for train steps
    loss_curve: <path_to_tb_or_csv>
    final_loss: <value>
    best_loss: <value>
    ppl: <value>
    steps: <int>
    duration_sec: <int>
    throughput_tokens_per_sec: <value>
    step_time_sec_avg: <value>
  metrics:
    <key>: <value>
  eval:
    # Only for eval steps
    datasets:
      - <name>
    results:
      <metric>: <value>
    script:
      path: <path>
      version: <commit_or_version>
  logs: <path>
  warnings:
    - <warn1>
    - <warn2>
  error: <error_message_if_failed>
  error_stack: <stack_trace_if_failed>
  retries: <int>

- step_id: 2
  name: ...
  stage: ...
  status: ...
  config: ...
  inputs:
    ...
  outputs:
    ...
  params:
    ...
  metrics:
    ...
  logs: ...
  warnings:
    ...
  error: ...

## Repro
- prepare: python3 scripts/prepare_exp.py -c <pipeline.py>
- run: python3 scripts/run.py -c <pipeline.py>
- env:
  DATAPOOL: <value>
  MEGATRON: <value>
  MINDSPEED: <value>
  ...

## Artifacts
- checkpoints:
  - path: <path>
    tag: <last|best|other>
    step: <int>
- hf_models:
  - <path>
- reports:
  - <path>

## Resources & Cost
- train_time_sec: <int>
- peak_vram_gb: <value>
- avg_vram_gb: <value>
- gpu_util_avg_pct: <value>

## Changes vs Previous
- previous_run_id: <run_id_or_n/a>
- config_diff: <path_or_inline_summary>

## Field Table (Checklist)
| section | field | description |
| --- | --- | --- |
| Summary | run_id | Run identifier |
| Summary | git_commit/git_dirty | Repo state |
| Environment | cuda_driver | Driver version |
| Dependencies | transformers/megatron/apex/te | Key versions |
| Data Copy | files_copied/files_skipped | Copy stats |
| Steps.training | loss/ppl/throughput/step_time | Training curves |
| Steps.eval | datasets/results/script | Eval context |
| Artifacts | last/best ckpt | Checkpoint lineage |
| Resources | peak_vram | Cost tracking |
| Changes | config_diff | Diff vs previous |

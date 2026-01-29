# step5: MindSpeed SFT training (extern script)
# EXTERN_SCRIPT: 直接运行可替代本 step 的全部逻辑
EXTERN_SCRIPT = "${MINDSPEED}/scripts/run_sft.sh"
EXTERN_SCRIPT_ARGS = ""

# 用于清理/归档的目录提示（由 run.py 读取）
CHECKPOINT_DIR = "${DATAPOOL_ROOT}/model/sft_checkpoints"


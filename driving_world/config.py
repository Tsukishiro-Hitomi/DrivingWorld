"""Default local model paths and shared video settings."""

from pathlib import Path

WORKSPACE = Path("/xs-fd-nas/workspace_zhaoxinrui")
H3_SCRIPT = WORKSPACE / "waymo-world-model/src/infer_h3.py"
H3_WEIGHTS = WORKSPACE / "models/MiniMax-H3-Ref2VA-BF16"
VLM_WEIGHTS = (
    WORKSPACE / "model_cache/models/Qwen--Qwen2.5-VL-7B-Instruct/snapshots/master"
)
FPS = 24

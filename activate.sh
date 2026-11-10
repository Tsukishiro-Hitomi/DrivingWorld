# Source this file from the repository root after selecting the H3 environment.
_drivingworld_root="$(pwd -P)"
if [ ! -f "$_drivingworld_root/agent_loop/dispatcher.py" ]; then
    printf '%s\n' 'Run source activate.sh from the DrivingWorld repository root.' >&2
    return 1
fi
if [ -f "$_drivingworld_root/.venv-h3/bin/activate" ]; then
    . "$_drivingworld_root/.venv-h3/bin/activate"
fi
export CROSSVIEW_ROOT="$_drivingworld_root"
export CROSSVIEW_MODEL_ROOT="${CROSSVIEW_MODEL_ROOT:-$CROSSVIEW_ROOT/models/MiniMax-H3-Ref2VA-BF16}"
export CROSSVIEW_WEIGHTS_ROOT="${CROSSVIEW_WEIGHTS_ROOT:-$CROSSVIEW_ROOT/weights}"
export CROSSVIEW_FL2VA_MODEL_ROOT="${CROSSVIEW_FL2VA_MODEL_ROOT:-$CROSSVIEW_ROOT/models/MiniMax-H3-FL2VA-BF16}"
export CROSSVIEW_QWEN_MODEL_ROOT="${CROSSVIEW_QWEN_MODEL_ROOT:-$CROSSVIEW_ROOT/models/Qwen3.8-27B}"
export PYTHONPATH="$CROSSVIEW_ROOT:$CROSSVIEW_ROOT/deps/DiffSynth-Studio:$CROSSVIEW_ROOT/deps/h3_runtime${PYTHONPATH:+:$PYTHONPATH}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export DIFFSYNTH_SKIP_DOWNLOAD=True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
unset _drivingworld_root

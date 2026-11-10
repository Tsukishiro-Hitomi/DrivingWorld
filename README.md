# DrivingWorld

Driving-video generation agents developed on the `wzy` branch.

| Entry | Function |
| --- | --- |
| `agent_loop.dispatcher` | Route text requests to an available agent |
| `agent_loop.planner` | Plan dependent tasks, execute them and review the combined result |
| `agent_loop.replace` | Replace an existing vehicle from frame zero; optional appearance image |
| `agent_loop.weather` | Edit atmosphere, lighting and camera-surface contamination |
| `agent_loop.insert` | Insert a new object; up to three local plan/prompt repairs |
| `agent_loop.full_text` | Generate an entire scene from text using native FL2VA/T2VA |

Each agent's `SKILL.md` contains its planning, prompt and review contract.
`ood_driving_experiment.py` and `configs/ood_driving_prompt.json` retain the earlier baseline/LLM/LLM+skill experiment entry; its source assets must be supplied separately.

## Environment

Reuse existing environments when available. The tested A800 setup uses **two separate environments**:

- H3: Python 3.10, PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, Transformers 4.57.6; dependencies in `requirements.txt`.
- Qwen worker: Python 3.11, PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121, Transformers 5.14.1; dependencies in `requirements-qwen.txt`.

For fresh environments, install a matching CUDA PyTorch/torchvision build first, then the corresponding requirements. Keep the worker separate to avoid replacing H3's Transformers version.

```bash
cd DrivingWorld
# Select the H3 environment first; .venv-h3 is activated automatically if present.
source activate.sh
export CROSSVIEW_QWEN_PYTHON="$PWD/.venv-qwen/bin/python" # or an existing Qwen environment
```

`activate.sh` adds the bundled DiffSynth snapshot and H3 adapter to `PYTHONPATH`. It does not install dependencies or download weights. SAM is needed only for replacement; `bash scripts/setup_sam.sh` checks out the pinned source, applies our tracking patch and installs it into the selected environment. Supply its separately licensed checkpoint through `CROSSVIEW_SAM_CHECKPOINT`.

## Weights and inputs

Weights, customer videos/images, outputs, caches and environments are not committed. Default locations:

```text
models/MiniMax-H3-Ref2VA-BF16/Ref2VA/
models/MiniMax-H3-FL2VA-BF16/FL2VA/
models/Qwen3.8-27B/
weights/MiniMax-H3_Ref2VA-LoRA-CrossView-Warp_v1_3500.safetensors
weights/minimax_h3_turbo_v4_step600_ema.safetensors
```

Override these with `CROSSVIEW_MODEL_ROOT`, `CROSSVIEW_FL2VA_MODEL_ROOT`, `CROSSVIEW_QWEN_MODEL_ROOT`, `CROSSVIEW_WEIGHTS_ROOT`, `CROSSVIEW_SAM_ROOT` and `CROSSVIEW_SAM_CHECKPOINT`. The legacy Qwen directory name and historical model labels are **not verified model identity**: the inspected checkpoint configuration declares `qwen3_5`. Confirm the source checkpoint before making version-specific claims.

The full-text weight preparer uses hf-mirror first, verifies Hub file identities and writes the deployment manifest/state/ready marker required by the executor:

```bash
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -m agent_loop.full_text.prepare_weights --destination "$CROSSVIEW_FL2VA_MODEL_ROOT"
```

## Run

```bash
# New scene: no source media.
python -m agent_loop.dispatcher run --agent auto \
  --instruction 'A severe blizzard; the camera is heavily blurred, lane markings are invisible, and traffic is recognizable mainly by hazard lights.' \
  --out outputs/full_text_blizzard

# Edit an existing compatible clip.
python -m agent_loop.dispatcher run --source inputs/source.mp4 \
  --instruction 'Insert one kitten entering from the right roadside.' --out outputs/kitten

# Combined edits: inspect the model-authored plan before explicit execution.
python -m agent_loop.planner plan --source inputs/source.mp4 \
  --instruction 'Replace the white vehicle ahead-left with a fire engine and make the weather heavy snow.' \
  --out outputs/composite
python -m agent_loop.planner execute --out outputs/composite
```

The current clip contract is 832×480, 124 frames at 24 FPS (~5.17 s). Text timelines do not change that duration. Image-guided replacement pauses for a host image-editor result; it is not fully unattended. Outer graph replanning and full-text repair are disabled; standalone weather allows one repair, and insertion allows up to three local repairs.

Runs preserve the model-authored plans/prompts, raw review responses, generated media and effective acceptance state. `PASS/FAIL/UNKNOWN` are evidence-bounded qualitative judgments; sparse images do not certify full-sequence continuity or metric GT. Invalid full-text reviews are retained and return `UNKNOWN`.

For CPU logic checks, run unittest discovery separately in `agent_loop/{planner,replace,weather,insert,full_text}/tests` with the H3 environment and `CUDA_VISIBLE_DEVICES=''`.

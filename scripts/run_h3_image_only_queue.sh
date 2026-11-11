#!/usr/bin/env bash
# Run remaining image-only H3 stress cases when a GPU has sufficient free VRAM.
set -euo pipefail

project_root="/xs-fd-nas/workspace_zhaoxinrui/driving-world"
python_bin="/root/miniconda3/envs/h3-ref2va/bin/python"
model_dir="/xs-fd-nas/workspace_zhaoxinrui/models/MiniMax-H3-Ref2VA-BF16"
experiment_dir="$project_root/experiments/h3_only_long_tail_001"
minimum_free_mib=60000

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
    awk -F, -v minimum="$minimum_free_mib" '$2 + 0 >= minimum {gsub(/ /, "", $1); print $1; exit}'
}

wait_for_gpu() {
  local gpu
  while true; do
    gpu="$(pick_gpu || true)"
    if [[ -n "$gpu" ]]; then
      printf '%s selected GPU %s\n' "$(date -Is)" "$gpu" >&2
      printf '%s' "$gpu"
      return
    fi
    printf '%s waiting for %s MiB free VRAM\n' "$(date -Is)" "$minimum_free_mib" >&2
    sleep 60
  done
}

run_case() {
  local identifier="$1"
  local seed="$2"
  local prompt="$3"
  local output="$experiment_dir/videos/$identifier.mp4"
  local log="$experiment_dir/logs/$identifier.log"
  if [[ -f "$output" ]]; then
    printf '%s already exists: %s\n' "$(date -Is)" "$output"
    return
  fi
  local gpu
  gpu="$(wait_for_gpu)"
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" "$project_root/infer_h3_image_only.py" \
    --model-dir "$model_dir" \
    --reference-image "$project_root/experiments/reference_h3_only_longtail_v1.png" \
    --output "$output" \
    --prompt "$prompt" \
    --height 256 --width 448 --num-frames 120 --steps 8 --seed "$seed" \
    >"$log" 2>&1
}

mkdir -p "$experiment_dir/videos" "$experiment_dir/logs"
run_case "rain_cut_in_pedestrian_brake" 301 "Realistic forward-facing dashcam footage. Within five seconds, rain begins and the road becomes wet while the camera remains fixed. The dark SUV from the right lane suddenly cuts left into the ego lane. At the same time a pedestrian in a yellow raincoat runs from the right curb into the roadway, and the white sedan directly ahead brakes sharply with red brake lights. Preserve all three actors and their distinct motions; no collision, no camera cut, no text or overlays."
run_case "night_glare_door_opening_dog" 303 "Transform this scene into a realistic wet night dashcam sequence with oncoming headlight glare and reflections, without changing the road layout. The white sedan ahead moves one lane left to avoid a stopped delivery van in the right lane. The delivery van's rear door swings open, and a small dog runs from the right sidewalk toward the road; the ego car brakes smoothly before reaching it. Keep vehicle identity, lane positions, door motion, dog motion, wet-road reflections, and stable camera coherent for the full video. No collision, no camera cut, no text or overlays."

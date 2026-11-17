# HyperFlow H3 Flash vs MiniMax-H3 Ref2VA: pilot evaluation

**Run ID:** hyperflow_flash_compare
**Date:** 2026-11-13 (Asia/Shanghai), 13:20–21:00
**Machine:** `di-20261023145159-mj47b`, 2× A800-SXM4-80GB (shared with other tenants), 463 GB RAM
**Question:** which of HyperFlow H3 Flash, the repo's 8-step Turbo LoRA, or VDN-H3 can replace the full
MiniMax-H3-Ref2VA-BF16 checkpoint in the proxy → H3 → Judge pipeline?

## Verdict (short)

Not a drop-in replacement on this hardware yet.

- **Speed:** 1.2–2.4× wall-clock, not the marketed 12.7×. Per denoising step Flash costs about the same as the full
  31B bf16 model on an A800 (≈48–50 s vs ≈54 s), because with a 5.2 s video reference the step is dominated by
  attention over ~14.8k tokens, not by the linear layers that the 14B INT8 compression shrinks.
- **Quality:** with the pipeline's own prompt Flash reaches the cut-in, but less decisively and visibly softer than
  the full model. The result is strongly prompt-sensitive: adding `<Picture 1>` / `<Video 1>` reference tokens made
  Flash anchor on the reference frame and skip the cut-in entirely at 4, 6 and 8 steps. Single seed, so this is a
  hypothesis, not a measured rate.
- **Integration:** Flash cannot generate fewer than 124 frames (5.2 s). The pipeline currently renders 2 s segments,
  so adopting Flash means switching segment length, or paying 2.6× the frames for a 2 s output.

Where Flash clearly wins: footprint (16 GB INT8 transformer vs 135 GB bf16 checkpoint), load time (25 s vs ~4 min),
and it is the only option on 24–32 GB consumer GPUs. On Hopper/Blackwell the INT8 kernels are the ones the 12.7×
claim was measured with; Ampere was not.

Neither alternative tested afterwards beats the plain dense model for this pipeline: the repo's 8-step Turbo LoRA,
once its PEFT scaling is fixed (0.0625 instead of 1.0), renders clean video at baseline speed but stops following the
proxy video; VDN-H3 has the right shape (8 NFE) but needs CUDA 12.9+ and this host's driver is CUDA 12.2.

## Setup

| | full model (baseline) | HyperFlow H3 Flash |
|---|---|---|
| checkpoint | `models/MiniMax-H3-Ref2VA-BF16` (31B bf16, 135 GB) | `models/hyperflow-h3-flash/ref2va_int8_convrot` (13.9B, W8A8 INT8, 16 GB) |
| text encoder / tokenizer / processor | in the checkpoint | reused from the local checkpoint (byte-identical shards) |
| vae / audio_vae / schedulers | in the checkpoint | `MiniMax/MiniMax-H3` from ModelScope into `models/MiniMax-H3-base` |
| sampler | diffusers default, CFG 1, 8 steps | fixed 4/6/8-step sigma grid, CFG 1 |
| env | `/root/miniconda3/envs/h3-ref2va` (torch 2.5.1) | `/dev/shm/hfenv` (python 3.12, torch 2.6.0+cu124, diffusers 0.43.0, transformers 5.22.2) |

Inputs, identical for every run: reference image `experiments/reference_h3_only_longtail_v1.png`, proxy video
`experiments/case_cut_in/proxy.mp4` (6 s grid render, right-lane SUV cutting into the ego lane at 2–5 s), 832×480,
124 aligned frames, 24 fps, seed 42. Prompt asks for the dark SUV to cut left into the ego lane around 2 s.
The full model received the pipeline's prompt; Flash received the same prompt plus `<Picture 1>` / `<Video 1>`
reference tokens (control run `flash_6step_h3prompt` repeats Flash with the plain prompt).

Reproduce: `run_h3_baseline.sh 8 h3_8step`, `run_flash_ref2va.sh <steps> <tag> [gpus]`.

## Measurements

End-to-end wall clock includes process start, weight load, denoising, and VAE/audio decode. GPU 1 only (GPU 0 is
occupied by other tenants; see "multi-GPU").

| run | steps | load | generate | wall | peak VRAM |
|---|---|---:|---:|---:|---:|
| full H3 bf16 | 8 | ≈240 s | 432 s (denoise) | 696 s | 78.1 GB |
| Flash INT8 | 4 | 24 s | 247.8 s | 292 s | 62.2 GB |
| Flash INT8 | 6 | 25 s | 348.5 s | 393 s | 62.2 GB |
| Flash INT8, pipeline prompt | 6 | ≈30 s | 336.1 s | 395 s | 62.2 GB |
| Flash INT8 | 8 | ≈30 s | 441.1 s | 577 s | 62.2 GB |
| Flash INT8, `int8_gemm=cublas` | 6 | 66 s | 337.8 s | 425 s | 62.2 GB |

- Per step: Flash ≈48–50 s, full H3 ≈54 s. Switching the INT8 GEMM backend moved the total by 3%, so the GEMM is
  not the bottleneck; the 14.8k-token attention over the reference video is.
- Wall-clock speedups vs the 8-step full model: **1.77×** (Flash 6), **2.38×** (Flash 4), **1.21×** (Flash 8).
- Multi-GPU (`--gpus 2`) could not be measured: GPU 0 carries ~74 GB from other tenants' processes, so the FSDP2
  shard load OOMs. On an idle pair of A800s this is the obvious next lever (the package supports 2/4-GPU context
  parallelism and VAE tiling).

## Quality

Frames at t=3.5 s, top to bottom: H3 8-step, Flash 6-step with reference tokens, Flash 6-step with the pipeline
prompt, Flash 8-step with reference tokens (`frames/compare_all_t3.5.png`).

- **H3 8-step:** the dark SUV is centered in the ego lane, cut-in complete; road markings and ego yaw read as one
  continuous maneuver. Crisp detail.
- **Flash, reference-token prompt (4, 6, 8 steps):** the SUV stays in the right lane for the whole clip, and the blue
  bus that sits far away in the reference image grows into the left lane. Appearance fidelity is high, motion
  fidelity is not. The token-rich prompt made the model treat the reference frame as the target state.
- **Flash, pipeline prompt (6 steps):** the SUV does move left toward the ego lane and ends roughly on the lane
  divider — direction and timing right, completion ambiguous. Closest Flash result to the baseline.
- **Sharpness:** Flash output is visibly softer (1.1–1.2 MB vs 6.5 MB for the same 124 frames at the same encoder
  settings; dashboard reflections and tree texture are the clearest losses). Appearance and layout stay plausible.
- **VLM judge:** `judge_event.py` (Qwen2.5-VL-7B object checker) scored *both* the baseline and Flash
  `not_observed`, including the baseline clip where the cut-in is plainly visible. The judge is not discriminative
  for this proxy-vs-render comparison as configured; the quality statements above rest on manual review of the
  contact sheets and frames. Reports: `reports/h3_8step.json`, `reports/flash_6step.json`.

Prompt sensitivity is itself a finding: whatever prompt adapter the pipeline gets must be validated against the
fallback-to-reference failure mode, not only against simple prompt following.

## Integration notes

1. **Minimum 124 frames (5.2 s).** `--num-frames` accepts 124–362 in 17n+5 steps. `driving_world/mvp.py` renders 2 s
   segments, so Flash would either force longer segments (fewer stitch boundaries, more compute per segment) or waste
   2.6× the frames per 2 s clip. Any adoption has to change the segment contract in `mvp.py`.
2. **Prompt convention.** `ref2va` references are addressed as `<Picture i>` / `<Video i>`; the current pipeline
   prompt has no such tokens. A prompt adapter belongs in `driving_world/scene_compiler.py` or `mvp.py`.
3. **Triton 3.2 patch.** The packaged INT8 kernels are defined inside a factory, so Triton 3.2 cannot resolve `tl`
   and raises `NameError: tl is not defined`. Mirroring `import triton` / `import triton.language as tl` at module
   level in `hyperflow_h3_flash/int8.py` fixes it (see `scripts/patch_int8.py`). Pinning torch ≥2.7 (Triton ≥3.3)
   may avoid the patch; unverified here.
4. **Audio.** Flash always generates a 32 kHz stereo track; the pipeline discards audio (`-an`), so its decode is
   pure overhead for us.
5. **Downloads.** `huggingface.co` is DNS-poisoned on this machine; `hf-mirror.com` works but throttles to ~1–5 MB/s.
   ModelScope (`VideoRebirthAI/hyperflow-h3-flash`, `MiniMax/MiniMax-H3`) delivered the same files at 15–50 MB/s.
   A 7 GB venv on the NAS took >25 min to install; `python3 -m venv /dev/shm/hfenv` installed in ~4 min.

## Recommendation

1. Do **not** swap the checkpoint yet. Run a multi-seed (≥5) A/B with the pipeline's real segment length, scored by
   human review plus a judge that is calibrated on this comparison; one pilot sample is not evidence about the
   motion-fidelity distribution, and the prompt control shows the failure mode is prompt-driven.
2. Re-measure on Hopper/Blackwell before drawing conclusions about Flash's speed. The 12.7× figure is a 4×B200
   number; on A800 the compression buys footprint, not step time.
3. If footprint or load time is the constraint (24–32 GB GPUs, many short-lived workers), Flash is already usable,
   with the segment-length and prompt-token changes above.
4. Do not use the repo's 8-step Turbo LoRA for proxy-conditioned work as it stands: even after correcting its
   scaling it drops the video conditioning (see below). Keep the dense model at 8 steps.
5. VDN-H3 is the most promising direction on paper (8 NFE, hybrid attention) but cannot run on this host: its stack
   needs CUDA 12.9+ (torch 2.13) and the driver here is 535 / CUDA 12.2. Re-test it on a Hopper/Blackwell host.

## 8-step Turbo LoRA (the repo's own checkpoint)

`models/h3-ref2va-turbo/minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors` (PEFT names, rank 128, metadata
alpha 8), loaded the way `waymo-world-model/outputs/turbo_worker.py` loads it (`pipe.load_lora(pipe.dit, path,
alpha=1.0, hotload=True)`, 8 steps, `flow_shift=6.0`).

| run | alpha | shift | wall | result |
|---|---:|---:|---:|---|
| `turbo_8step` (repo recipe) | 1.0 | 6.0 | 634 s | **dark noise, unusable** |
| `lora_alpha00625` | 8/128 | 6.0 | 838 s | clean, cut-in missing |
| `lora_alpha00625_defaultshift` | 8/128 | 12.0 | 711 s | clean, cut-in missing |
| baseline, no LoRA | — | 12.0 | 696 s (8 steps) / 1752 s (30 steps) | cut-in complete |

- **The scaling is wrong in the repo wrapper.** PEFT applies `alpha / rank` = 8/128 = 0.0625; the wrapper passes
  `alpha=1.0`, 16× the trained scale, which destroys the sample. With 0.0625 the LoRA renders normally, so the
  "LoRA noise" that `waymo-world-model/outputs/non_waymo_40_v1/DATASET_QC.md` excludes from the dataset is a
  scaling bug, not a broken checkpoint.
- **Corrected, it still loses the proxy conditioning.** At t=3.5 s the baseline puts the dark SUV in the ego lane
  (both 8 and 30 steps); all three LoRA runs leave it in the right lane and let the ego car pass it. The DMD/turbo
  adapter was distilled for text/keyframe conditioning, so the reference-video rows stop steering motion.
- **Speed is identical by construction.** Same 31B transformer, same 8 forward passes: 55–57 s/step with the LoRA
  vs 52.5 s/step without, i.e. the LoRA costs ~5% per step and saves nothing at equal step count. It only pays off if
  8 steps were otherwise unaffordable — and the dense model already produces the correct event at 8 steps.

## VDN-H3 (weights present, blocked by the driver)

`vdn-h3/vdn-minimax-h3` (82 GB, `ckpts/` with `h3-base`, `stage-b-step-2000`, `stage-dmd-step-250` — the 8-NFE DMD
checkpoint) is on disk, and it is the closest thing to what we want: 8 NFE, hybrid window-softmax + linear attention,
reported 32.7 → 11.2 s/NFE on one H200.

It cannot run on this host:

- The repo pins torch 2.13 / cu129 (`pyproject.toml`) and needs `torch.nn.attention.varlen`,
  `torch._dynamo.config.fail_on_recompile_limit_hit`. The conda env `vdn-h3` has torch 2.5.1, so `diffusers 0.42.1`
  fails at import (`torch.accelerator` missing).
- Installing torch 2.13 resolves to cu130, which needs a ≥580 driver; this box runs 535 (CUDA 12.2), so
  `torch.cuda.is_available()` is False. torch 2.10+cu128 does see the GPUs, but transformers 5.15 then upgrades
  torch back to 2.15+cu132 and CUDA disappears again.
- Also structural: VDN's "Ref2VA-like" mode takes **images only** (`encode_keyframes.py --refs`), rendered by the
  FL2VA weights. There is no reference-video conditioning, so even on the right hardware it could not consume the
  proxy video the pipeline depends on — only the first frame plus the prompt.

Next step if we want this measured: run it on an H200/B200 host with driver ≥ 580, encode a prompt with
`--refs` (first frame), render 124 frames at 8 NFE, and compare that image-only clip against an image-only dense
baseline rather than against the proxy-conditioned one.

## Artifacts

```
experiments/hyperflow_flash_compare/
├── videos/            h3_{8,30}step, flash_{4,6,8}step, flash_6step_{cublas,h3prompt},
│                      turbo_8step (noise), lora_alpha00625, lora_alpha00625_defaultshift
├── contact_sheets/    12-frame sheets per clip
├── frames/            t=1.5 s and t=3.5 s stills, compare_all_t3.5.png, compare_lora_t3.5.png
├── reports/           judge_event.py outputs
├── logs/              run logs (timings, memory)
├── events/cut_in_c0.json
├── scripts/           run_flash.py (Triton patch), infer_h3_lora.py (PEFT scaling), patch_int8.py,
│                      vdn_{encode,infer}.py + vdn_shim.py, download/setup helpers
├── run_h3_baseline.sh
├── run_h3_turbo.sh          (repo LoRA recipe, alpha 1.0)
├── run_lora_alpha.sh        (LoRA with explicit alpha / shifts)
├── run_lora_alpha2.sh       (LoRA, default shifts)
└── run_flash_ref2va.sh
```

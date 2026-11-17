"""Time a 90-frame MiniMax-H3 render end to end, split into denoise / video-VAE / audio-VAE
/ muxing. Arms are rendered in one process so the 72 GB checkpoint is loaded once.

    python bench90.py --checkpoint null --config configs/inference/50nfe_tuned.yaml \
        --arms '{"name": "dense_480p_8nfe", "latent_h": 30, "latent_w": 52, "steps": 8}' ...
"""
import argparse
import json
import os
import sys
import time

REPO = "/xs-fd-nas/workspace_zhaoxinrui/vdn-h3/vdn-minimax-h3"
sys.path.insert(0, "/root")
import accel_shim  # noqa: E402

accel_shim.install()
sys.path.insert(0, REPO)

import torch  # noqa: E402

# torch 2.10's varlen_attn predates the `scale` keyword this vendored stack passes
# (torch 2.13 has it). softmax(qk * s) == softmax((q * s) k), so fold the scale into q.
import torch.nn.attention.varlen as _varlen  # noqa: E402

_orig_varlen_attn = _varlen.varlen_attn


def _varlen_attn_with_scale(q, k, v, cu_q, cu_k, max_q, max_k, *, scale=None, **kw):
    if scale is not None:
        q = q * scale
    return _orig_varlen_attn(q, k, v, cu_q, cu_k, max_q, max_k, **kw)


_varlen.varlen_attn = _varlen_attn_with_scale

from src.config import load_config, resolved_dict  # noqa: E402
from src.config.inference import (InferenceConfig, validate_ablation,  # noqa: E402
                                  validate_kernels, validate_single_process)
from src.inference.render import PIXEL_MEAN, PIXEL_STD, generate_latents  # noqa: E402
from src.inference.utils.assemble import build_inference_model  # noqa: E402
from src.inference.utils.prompt_cache import load_prompt  # noqa: E402
import src.inference.render as render  # noqa: E402
from diffusers.utils.export_utils import encode_video  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default="null")
    p.add_argument("--frames", type=int, default=90)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=float, default=24.0)
    p.add_argument("--prompt-file", default="prompts/example_0.pt")
    p.add_argument("--outdir", default="/dev/shm/bench90")
    p.add_argument("--json", required=True)
    p.add_argument("--arms", default="[]", help="JSON list of arms")
    p.add_argument("--arms-file", help="path to a JSON list of arms (overrides --arms)")
    p.add_argument("--skip-decode", action="store_true")
    p.add_argument("dotlist", nargs="*")
    return p.parse_args()


@torch.no_grad()
def decode_split(latents, audio_latents, vae, audio_vae, out_path, device, fps,
                 with_audio=True):
    """decode_and_save with a timer around each stage."""
    marks = {}
    t0 = time.perf_counter()
    vae.to(device)
    latents_mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        video = vae.decode(latents * latents_std + latents_mean, return_dict=False)[0]
    torch.cuda.synchronize(device)
    marks["video_vae"] = time.perf_counter() - t0
    pixel_mean = torch.tensor(PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor(PIXEL_STD, device=device).view(1, -1, 1, 1, 1)
    video = (video.float() * pixel_std + pixel_mean).clamp(0, 1)
    t1 = time.perf_counter()
    audio = None
    if with_audio:
        audio_vae.to(device)
        audio_latents_mean = torch.tensor(audio_vae.config.latents_mean, device=device).view(1, -1, 1)
        audio_latents_std = torch.tensor(audio_vae.config.latents_std, device=device).view(1, -1, 1)
        audio = audio_vae.decode(
            audio_latents * audio_latents_std + audio_latents_mean, return_dict=False)[0]
        audio = audio.float().permute(1, 0, 2)[0]
        torch.cuda.synchronize(device)
    marks["audio_vae"] = time.perf_counter() - t1
    t2 = time.perf_counter()
    frames = (video[0].permute(1, 2, 3, 0) * 255).round().to(torch.uint8).cpu()
    marks["numpy"] = time.perf_counter() - t2
    t3 = time.perf_counter()
    if with_audio:
        encode_video(frames, fps=fps, output_path=out_path,
                     audio=audio.cpu(), audio_sample_rate=audio_vae.config.sampling_rate)
    else:
        encode_video(frames, fps=fps, output_path=out_path, audio=None)
    marks["encode"] = time.perf_counter() - t3
    return marks


def main():
    args = parse_args()
    os.chdir(REPO)
    arms = json.loads(args.arms)
    if args.arms_file:
        with open(args.arms_file) as f:
            arms = json.load(f)
    device = "cuda:0"
    torch.set_grad_enabled(False)

    argv = ["--config", args.config] + args.dotlist + [
        f"checkpoint={args.checkpoint}", f"render.num_frames={args.frames}",
        f"render.warmup_steps=0", f"render.seed={args.seed}",
        f"render.prompt_file={args.prompt_file}",
        f"render.out={args.outdir}/unused.mp4",
    ]
    cfg = load_config(InferenceConfig, argv=argv,
                      extra_validators=[validate_ablation, validate_kernels,
                                        validate_single_process])
    started = time.perf_counter()
    model = build_inference_model(cfg, device)
    load_seconds = time.perf_counter() - started
    prompt_embeds, text_token_tags, conditions = load_prompt(cfg.render.prompt_file, device)
    print(f"loaded in {load_seconds:.1f}s; {len(arms)} arms", flush=True)

    record = {"name": args.name, "checkpoint": cfg.checkpoint, "gpu": torch.cuda.get_device_name(device),
              "load_seconds": load_seconds, "config": resolved_dict(cfg), "arms": {}}

    for arm in arms:
        name = arm["name"]
        render.LATENT_H, render.LATENT_W = arm["latent_h"], arm["latent_w"]
        torch.cuda.reset_peak_memory_stats(device)
        step_seconds = []
        t0 = time.perf_counter()
        latents, audio_latents = generate_latents(
            model.transformer, prompt_embeds, text_token_tags, args.frames, arm["steps"],
            args.seed, device, video_shift=cfg.render.video_shift,
            audio_shift=cfg.render.audio_shift, step_seconds=step_seconds,
            conditions=conditions)
        torch.cuda.synchronize(device)
        denoise = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated(device) / 2**30
        latent_path = os.path.join(args.outdir, f"{name}.pt")
        os.makedirs(args.outdir, exist_ok=True)
        torch.save({"latents": latents.cpu(), "audio": audio_latents.cpu(),
                    "frames": args.frames}, latent_path)
        arm["latents"] = latent_path
        arm["denoise"] = denoise
        arm["s_per_nfe"] = denoise / len(step_seconds)
        arm["step_seconds"] = step_seconds
        arm["denoise_peak_gib"] = peak
        print(f"[{name}] denoise {denoise:.2f}s ({arm['s_per_nfe']:.2f}s/NFE), peak {peak:.1f} GiB",
              flush=True)
        record["arms"][name] = arm

    if not args.skip_decode:
        # Give the transformer's 62 GiB back before the decoders come up.
        model.transformer.to("cpu")
        torch.cuda.empty_cache()
        for arm in arms:
            name = arm["name"]
            payload = torch.load(arm["latents"], map_location="cpu")
            latents, audio_latents = payload["latents"].to(device), payload["audio"].to(device)
            out = os.path.join(args.outdir, f"{name}.mp4")
            torch.cuda.reset_peak_memory_stats(device)
            marks = decode_split(latents, audio_latents, model.vae, model.audio_vae,
                                 out, device, args.fps, with_audio=True)
            total = sum(marks.values())
            arm["decode"] = marks
            arm["decode_total"] = total
            arm["decode_peak_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
            print(f"[{name}] decode: {total:.2f}s {marks}", flush=True)
            arm["end_to_end"] = arm["denoise"] + total
            print(f"[{name}] end-to-end denoise+decode {arm['end_to_end']:.2f}s", flush=True)

    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(record, f, indent=2, sort_keys=True)
    print(f"wrote {args.json}", flush=True)


if __name__ == "__main__":
    main()

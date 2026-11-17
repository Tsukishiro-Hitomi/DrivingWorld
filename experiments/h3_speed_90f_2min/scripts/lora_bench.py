"""A/B a step-distillation LoRA against the dense Ref2VA path, one loaded pipeline.

Arms differ only in (LoRA on/off, steps). Same prompt embedding, same three keyframe
references, same 90 frames at 480p; denoise/decode are timed separately.
"""
import argparse
import json
import time
from pathlib import Path

from PIL import Image
import torch

from h3_ref2va.cli import aligned_frames, model_configs
from h3_ref2va.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline
from h3_ref2va.utils.data.audio_video import read_video_audio, write_video_audio

PROMPT = (
    "Continue the exact photorealistic daytime driving scene from the reference image. "
    "A blue SUV in the right lane moves into the ego lane and stays ahead. Keep the road, "
    "asphalt, sky, lighting and traffic photorealistic. The attached video is only a motion "
    "reference. Do not show diagrams, captions, CGI, boxes, or overlays."
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--reference-image", required=True)
    p.add_argument("--reference-video", required=True)
    p.add_argument("--lora", default=None)
    p.add_argument("--frames", type=int, default=90)
    p.add_argument("--outdir", default="/dev/shm/bench90")
    p.add_argument("--json", required=True)
    p.add_argument("--arms", default="[]")
    p.add_argument("--arms-file")
    p.add_argument("--vram-limit", type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    args.model_dir = Path(args.model_dir)
    arms = json.loads(args.arms)
    if args.arms_file:
        with open(args.arms_file) as f:
            arms = json.load(f)
    num_frames = aligned_frames(args.frames)

    configs, processor, vram_limit = model_configs(args.model_dir, "cpu")
    pipe = MiniMaxH3Pipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device="cuda", model_configs=configs,
        processor_config=processor, vram_limit=args.vram_limit or vram_limit)

    emb = pipe.export_text_embedding(PROMPT)
    frames, _, _ = read_video_audio(
        args.reference_video, 480, 832, num_frames, fps=24, audio_sample_rate=None)
    picks = [frames[0], frames[len(frames) // 2], frames[-1]]
    refs = [{"type": "image", "image": Image.open(args.reference_image).convert("RGB")}]
    refs += [{"type": "image", "image": f} for f in picks]

    record = {"lora": args.lora, "frames": num_frames, "gpu": torch.cuda.get_device_name(0),
              "arms": {}}
    loaded_alpha = None
    for arm in arms:
        alpha = float(arm.get("alpha", 1.0)) if arm.get("lora") else None
        if alpha is not None and alpha != loaded_alpha:
            if loaded_alpha is not None:
                pipe.clear_lora()
            t0 = time.perf_counter()
            pipe.load_lora(pipe.dit, args.lora, alpha=alpha, hotload=True)
            torch.cuda.synchronize()
            record.setdefault("lora_load_seconds", {})[str(alpha)] = time.perf_counter() - t0
            loaded_alpha = alpha
        elif alpha is None and loaded_alpha is not None:
            pipe.clear_lora()
            loaded_alpha = None

        steps = []
        orig_cfg = pipe.cfg_guided_model_fn

        def timed_cfg(*a, **k):
            t0 = time.perf_counter()
            out = orig_cfg(*a, **k)
            torch.cuda.synchronize()
            steps.append(time.perf_counter() - t0)
            return out

        pipe.cfg_guided_model_fn = timed_cfg
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        video, audio = pipe(
            prompt=PROMPT, height=arm.get("height", 480), width=arm.get("width", 832),
            num_frames=num_frames,
            num_inference_steps=arm["steps"], seed=42, references=refs,
            ref_image_short_edge=480, text_embedding=emb)
        torch.cuda.synchronize()
        total = time.perf_counter() - t0
        pipe.cfg_guided_model_fn = orig_cfg

        out = f"{args.outdir}/{arm['name']}.mp4"
        Path(args.outdir).mkdir(parents=True, exist_ok=True)
        write_video_audio(video, audio, out, fps=24, audio_sample_rate=pipe.audio_vae.sample_rate)
        denoise = sum(steps)
        entry = dict(arm)
        entry.update({"total": total, "denoise": denoise,
                      "s_per_nfe": denoise / max(1, len(steps)), "steps_seconds": steps,
                      "peak_gib": torch.cuda.max_memory_allocated() / 2**30, "out": out})
        record["arms"][arm["name"]] = entry
        print(f"[{arm['name']}] total {total:.1f}s denoise {denoise:.1f}s "
              f"({entry['s_per_nfe']:.2f}s/NFE) peak {entry['peak_gib']:.1f} GiB", flush=True)

    with open(args.json, "w") as f:
        json.dump(record, f, indent=2, sort_keys=True)
    print(f"wrote {args.json}", flush=True)


if __name__ == "__main__":
    main()

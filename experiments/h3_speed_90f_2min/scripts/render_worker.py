"""Persistent Ref2VA worker prototype: one process, one prompt encode, resident render set.

VRAM budget on an 80 GB card (bf16):
  transformer 62 GB + video_vae 9.8 GB + audio_vae 0.6 GB = 72.4 GB resident,
  text encoder 63 GB is loaded once for the prompt, then handed back to the CPU.

Sequence:
  1. load the pipeline (weights on CPU, stock vram management)
  2. onload text_encoder -> export_text_embedding -> onload the render set (evicts it)
  3. replace load_models_to_device with a no-op: nothing is ever swapped again
  4. warm up one 1-step 90-frame request (kernel autotune + VAE first decode)
  5. render two full 8-step requests and report each latency and the peak VRAM
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
    p.add_argument("--frames", type=int, default=90)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--requests", type=int, default=2)
    p.add_argument("--outdir", default="/dev/shm/bench90")
    p.add_argument("--json", required=True)
    p.add_argument("--vram-limit", type=float, default=78.5)
    return p.parse_args()


def main():
    args = parse_args()
    args.model_dir = Path(args.model_dir)
    num_frames = aligned_frames(args.frames)
    device = "cuda"

    t0 = time.perf_counter()
    configs, processor, vram_limit = model_configs(args.model_dir, "cpu")
    # The render set must really live on the GPU: with the stock config onload_device is
    # "cpu", so load_models_to_device() only ever moves weights back to the CPU and every
    # forward streams blocks over PCIe. The text encoder stays on CPU -- it is used once.
    for cfg in configs:
        if any("text_encoder" in str(p) for p in cfg.path):
            continue
        cfg.onload_device = "cuda"
        cfg.offload_device = "cuda"
        cfg.preparing_device = "cuda"
        cfg.computation_device = "cuda"
    pipe = MiniMaxH3Pipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device=device, model_configs=configs,
        processor_config=processor, vram_limit=args.vram_limit)
    load_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    pipe.load_models_to_device(["text_encoder"])
    emb = pipe.export_text_embedding(PROMPT)
    torch.cuda.synchronize()
    text_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    pipe.load_models_to_device(["transformer", "video_vae", "audio_vae"])
    torch.cuda.synchronize()
    resident_seconds = time.perf_counter() - t0
    pipe.load_models_to_device = lambda names: None  # nothing swaps from here on
    resident_alloc = torch.cuda.memory_allocated() / 2**30
    print(f"resident set {resident_alloc:.1f} GiB", flush=True)

    frames, _, _ = read_video_audio(
        args.reference_video, 480, 832, num_frames, fps=24, audio_sample_rate=None)
    picks = [frames[0], frames[len(frames) // 2], frames[-1]]
    refs = [{"type": "image", "image": Image.open(args.reference_image).convert("RGB")}]
    refs += [{"type": "image", "image": f} for f in picks]

    def render(steps, out):
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        video, audio = pipe(
            prompt=PROMPT, height=480, width=832, num_frames=num_frames,
            num_inference_steps=steps, seed=42, references=refs,
            ref_image_short_edge=480, text_embedding=emb)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t
        write_video_audio(video, audio, out, fps=24,
                          audio_sample_rate=pipe.audio_vae.sample_rate)
        return seconds, torch.cuda.max_memory_allocated() / 2**30

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    warm_seconds, warm_peak = render(1, f"{args.outdir}/worker_warmup.mp4")
    print(f"warmup 1 step: {warm_seconds:.1f}s peak {warm_peak:.1f} GiB", flush=True)

    runs = []
    for i in range(args.requests):
        seconds, peak = render(args.steps, f"{args.outdir}/worker_request{i}.mp4")
        runs.append({"seconds": seconds, "peak_gib": peak})
        print(f"request {i}: {seconds:.1f}s @ {args.steps} steps, peak {peak:.1f} GiB",
              flush=True)

    record = {
        "load_seconds": load_seconds, "text_seconds": text_seconds,
        "resident_seconds": resident_seconds, "resident_alloc_gib": resident_alloc,
        "warmup_seconds": warm_seconds,
        "warmup_peak_gib": warm_peak, "runs": runs, "frames": num_frames,
        "steps": args.steps, "gpu": torch.cuda.get_device_name(0),
    }
    with open(args.json, "w") as f:
        json.dump(record, f, indent=2, sort_keys=True)
    print(f"wrote {args.json}", flush=True)


if __name__ == "__main__":
    main()

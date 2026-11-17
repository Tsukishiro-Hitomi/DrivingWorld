"""Time the production Ref2VA path (proxy video + first-frame image) on an A800.

Three arms share one loaded pipeline:
  ref480_8nfe  reference video encoded at 480 short edge, 8 NFE
  ref768_8nfe  reference video at the wrapper default 768 short edge, 8 NFE

Per-step time comes from wrapping `cfg_guided_model_fn`; decode time from wrapping
`video_vae.decode_video`; the rest is text/reference encoding.
"""
import argparse
import json
import time

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
    p.add_argument("--outdir", default="/dev/shm/bench90")
    p.add_argument("--json", required=True)
    p.add_argument("--arms", default="[]", help="JSON list of {name, steps, ref_short_edge}")
    p.add_argument("--arms-file")
    p.add_argument("--offload", default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    arms = json.loads(args.arms)
    if args.arms_file:
        with open(args.arms_file) as f:
            arms = json.load(f)
    num_frames = aligned_frames(args.frames)

    started = time.perf_counter()
    configs, processor, vram_limit = model_configs(args.model_dir, args.offload)
    pipe = MiniMaxH3Pipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device="cuda", model_configs=configs,
        processor_config=processor, vram_limit=vram_limit)
    load_seconds = time.perf_counter() - started
    print(f"loaded in {load_seconds:.1f}s, vram_limit={vram_limit}", flush=True)

    record = {"load_seconds": load_seconds, "frames": num_frames, "gpu": torch.cuda.get_device_name(0),
              "arms": {}}
    for arm in arms:
        frames, _, _ = read_video_audio(
            args.reference_video, 480, 832, num_frames, fps=24, audio_sample_rate=None)
        references = [
            {"type": "image", "image": Image.open(args.reference_image).convert("RGB")},
            {"type": "video", "video": frames},
        ]
        steps, decode = [], []
        orig_cfg = pipe.cfg_guided_model_fn
        orig_decode = pipe.video_vae.decode_video

        def timed_cfg(*a, **k):
            t0 = time.perf_counter()
            out = orig_cfg(*a, **k)
            torch.cuda.synchronize()
            steps.append(time.perf_counter() - t0)
            return out

        def timed_decode(*a, **k):
            t0 = time.perf_counter()
            out = orig_decode(*a, **k)
            torch.cuda.synchronize()
            decode.append(time.perf_counter() - t0)
            return out

        pipe.cfg_guided_model_fn = timed_cfg
        pipe.video_vae.decode_video = timed_decode
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        video, audio = pipe(
            prompt=PROMPT, height=480, width=832, num_frames=num_frames,
            num_inference_steps=arm["steps"], seed=42, references=references,
            ref_video_short_edge=arm["ref_short_edge"])
        torch.cuda.synchronize()
        total = time.perf_counter() - t0
        pipe.cfg_guided_model_fn = orig_cfg
        pipe.video_vae.decode_video = orig_decode
        out = f"{args.outdir}/{arm['name']}.mp4"
        write_video_audio(video, audio, out, fps=24, audio_sample_rate=pipe.audio_vae.sample_rate)
        entry = dict(arm)
        entry.update({
            "total": total,
            "encode": total - sum(steps) - sum(decode),
            "denoise": sum(steps),
            "steps_seconds": steps,
            "decode": sum(decode),
            "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
            "out": out,
        })
        record["arms"][arm["name"]] = entry
        print(f"[{arm['name']}] total {total:.1f}s = encode {entry['encode']:.1f} + denoise "
              f"{entry['denoise']:.1f} ({entry['denoise'] / len(steps):.1f}s/NFE) + decode "
              f"{entry['decode']:.1f}s, peak {entry['peak_gib']:.1f} GiB", flush=True)

    with open(args.json, "w") as f:
        json.dump(record, f, indent=2, sort_keys=True)
    print(f"wrote {args.json}", flush=True)


if __name__ == "__main__":
    main()

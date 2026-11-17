"""Ref2VA speed arms at 90 frames / 480p on one A800, one loaded pipeline.

Arms differ only in what is fed as the reference and whether the prompt embedding is
reused, so the deltas are the cost of the reference rows, not of the checkpoint:

  video90_cached   full 90-frame proxy reference, text embedding reused
  video24_cached   the same proxy subsampled to 24 frames (1 fps)
  image3_cached    first / middle / last proxy frame as three image references
  video90_coldtext the 90-frame reference with the prompt encoded in-call (production)

Per-unit time (prompt embedder, reference encoder, packed sequence, denoise, decode) is
collected by wrapping `unit_runner`, `cfg_guided_model_fn` and `decode_video`.
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
    p.add_argument("--outdir", default="/dev/shm/bench90")
    p.add_argument("--json", required=True)
    p.add_argument("--arms", default="[]")
    p.add_argument("--arms-file")
    p.add_argument("--offload", default="cpu")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    return p.parse_args()


def main():
    args = parse_args()
    arms = json.loads(args.arms)
    if args.arms_file:
        with open(args.arms_file) as f:
            arms = json.load(f)
    num_frames = aligned_frames(args.frames)
    args.model_dir = Path(args.model_dir)

    started = time.perf_counter()
    configs, processor, vram_limit = model_configs(args.model_dir, args.offload)
    pipe = MiniMaxH3Pipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device="cuda", model_configs=configs,
        processor_config=processor, vram_limit=vram_limit)
    load_seconds = time.perf_counter() - started
    print(f"loaded in {load_seconds:.1f}s", flush=True)

    # Prompt embedding, once: every arm that sets reuse_text_embedding skips this.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    emb = pipe.export_text_embedding(PROMPT)
    torch.cuda.synchronize()
    text_embed_seconds = time.perf_counter() - t0
    print(f"text embedding {text_embed_seconds:.1f}s", flush=True)

    frames, _, _ = read_video_audio(
        args.reference_video, args.height, args.width, num_frames, fps=24,
        audio_sample_rate=None)
    image = Image.open(args.reference_image).convert("RGB")

    record = {"load_seconds": load_seconds, "text_embed_seconds": text_embed_seconds,
              "frames": num_frames, "gpu": torch.cuda.get_device_name(0), "arms": {}}
    for arm in arms:
        refs = [{"type": "image", "image": image}]
        if arm["reference"] == "video90":
            refs.append({"type": "video", "video": frames})
        elif arm["reference"] == "video24":
            # 24 frames spread over the whole clip, not the first 24 of a stride-3 walk.
            idx = [round(i * (len(frames) - 1) / 23) for i in range(24)]
            refs.append({"type": "video", "video": [frames[i] for i in idx]})
        elif arm["reference"] == "image3":
            picks = [frames[0], frames[len(frames) // 2], frames[-1]]
            refs += [{"type": "image", "image": f} for f in picks]
        else:
            raise ValueError(arm["reference"])

        steps, decodes, units = [], [], {}
        orig_cfg = pipe.cfg_guided_model_fn
        orig_decode = pipe.video_vae.decode_video
        orig_runner = pipe.unit_runner
        orig_load = pipe.load_models_to_device
        onload = []

        def timed_load(*a, **k):
            t0 = time.perf_counter()
            out = orig_load(*a, **k)
            torch.cuda.synchronize()
            onload.append(time.perf_counter() - t0)
            return out

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
            decodes.append(time.perf_counter() - t0)
            return out

        def timed_runner(unit, *a, **k):
            name = type(unit).__name__.replace("MiniMaxH3Unit_", "")
            t0 = time.perf_counter()
            out = orig_runner(unit, *a, **k)
            torch.cuda.synchronize()
            units[name] = units.get(name, 0.0) + time.perf_counter() - t0
            return out

        pipe.cfg_guided_model_fn = timed_cfg
        pipe.video_vae.decode_video = timed_decode
        pipe.unit_runner = timed_runner
        pipe.load_models_to_device = timed_load
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        video, audio = pipe(
            prompt=PROMPT, height=args.height, width=args.width, num_frames=num_frames,
            num_inference_steps=arm["steps"], seed=42, references=refs,
            ref_video_short_edge=arm.get("ref_short_edge", 480),
            ref_image_short_edge=arm.get("ref_image_short_edge", 480),
            text_embedding=emb if arm.get("reuse_text_embedding", True) else None)
        torch.cuda.synchronize()
        total = time.perf_counter() - t0
        pipe.cfg_guided_model_fn = orig_cfg
        pipe.video_vae.decode_video = orig_decode
        pipe.unit_runner = orig_runner
        pipe.load_models_to_device = orig_load

        out = f"{args.outdir}/{arm['name']}.mp4"
        write_video_audio(video, audio, out, fps=24, audio_sample_rate=pipe.audio_vae.sample_rate)
        entry = dict(arm)
        entry.update({"total": total, "denoise": sum(steps),
                      "denoise_per_nfe": sum(steps) / max(1, len(steps)),
                      "steps_seconds": steps, "decode": sum(decodes),
                      "units": units, "onload": sum(onload), "onload_calls": len(onload), "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                      "out": out})
        record["arms"][arm["name"]] = entry
        print(f"[{arm['name']}] total {total:.1f}s denoise {entry['denoise']:.1f}s "
              f"({entry['denoise_per_nfe']:.1f}s/NFE) decode {entry['decode']:.1f}s "
              f"units { {k: round(v, 1) for k, v in units.items()} } "
              f"peak {entry['peak_gib']:.1f} GiB", flush=True)

    with open(args.json, "w") as f:
        json.dump(record, f, indent=2, sort_keys=True)
    print(f"wrote {args.json}", flush=True)


if __name__ == "__main__":
    main()

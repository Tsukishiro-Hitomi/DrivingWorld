#!/usr/bin/env python3
"""Run MiniMax H3 from a first-frame image and text prompt only."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
import torch

from h3_ref2va.cli import aligned_frames, model_configs
from h3_ref2va.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline
from h3_ref2va.utils.data.audio_video import write_video_audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference-image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=120)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offload", choices=("none", "cpu", "disk"), default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.num_frames = aligned_frames(args.num_frames)
    if not args.reference_image.is_file():
        raise FileNotFoundError(args.reference_image)
    configs, processor, vram_limit = model_configs(args.model_dir, args.offload)
    pipe = MiniMaxH3Pipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=configs,
        processor_config=processor,
        vram_limit=vram_limit,
    )
    video, audio = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        seed=args.seed,
        references=[{"type": "image", "image": Image.open(args.reference_image).convert("RGB")}],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_video_audio(
        video,
        audio,
        str(args.output),
        fps=24,
        audio_sample_rate=pipe.audio_vae.sample_rate,
    )
    print(f"Wrote {args.num_frames} H3 image-only frames to {args.output}")


if __name__ == "__main__":
    main()

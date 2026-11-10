#!/usr/bin/env python3
"""Generate and assemble proxy-conditioned H3 video segments.

Event assessment deliberately lives in ``driving_world.judge``. Keeping this
module generation-only prevents a candidate from being selected by a generic
review that does not know the requested event contract.
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import imageio.v3 as iio
import imageio_ffmpeg
from PIL import Image

from .config import FPS, H3_SCRIPT, H3_WEIGHTS


def run(command):
    command = list(map(str, command))
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def extract_last_frame(video, destination):
    last = None
    for last in iio.imiter(video, plugin="FFMPEG"):
        pass
    if last is None:
        raise ValueError(f"No frames in {video}")
    Image.fromarray(last).save(destination)


def generate_segment(args, index, segment, reference, duration, ffmpeg):
    """Generate one segment; a Decider selects retries in a later pipeline stage."""
    proxy = segment / "proxy.mp4"
    start = index * args.segment_seconds
    run(
        [
            ffmpeg,
            "-y",
            "-ss",
            start,
            "-i",
            args.proxy,
            "-vf",
            f"tpad=stop_mode=clone:stop_duration={args.segment_seconds}",
            "-t",
            args.segment_seconds,
            "-an",
            "-r",
            FPS,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            proxy,
        ]
    )

    candidate = segment / "candidate.mp4"
    run(
        [
            sys.executable,
            H3_SCRIPT,
            "--model-dir",
            H3_WEIGHTS,
            "--reference-image",
            reference,
            "--reference-video",
            proxy,
            "--output",
            candidate,
            "--prompt",
            args.prompt,
            "--height",
            args.height,
            "--width",
            args.width,
            "--num-frames",
            max(22, round(args.segment_seconds * FPS)),
            "--steps",
            args.steps,
            "--seed",
            args.seed + index * 100,
        ]
    )
    selected = segment / "selected.mp4"
    run(
        [
            ffmpeg,
            "-y",
            "-i",
            candidate,
            "-t",
            duration,
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            selected,
        ]
    )
    return {
        "segment": index,
        "video": str(selected),
        "candidate": str(candidate),
        "outcome": "generated_unjudged",
    }


def assemble(args):
    if not args.proxy.is_file() or not args.reference_image.is_file():
        raise FileNotFoundError("Proxy video and reference image must exist")
    if args.segment_seconds <= 0 or args.steps < 1:
        raise ValueError("segment duration and steps must be positive")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    duration = iio.immeta(args.proxy, plugin="FFMPEG").get("duration")
    if not duration or not math.isfinite(duration) or duration <= 0:
        raise ValueError("Cannot determine proxy duration")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    selected = []
    reference = args.reference_image.resolve()
    for index in range(math.ceil(duration / args.segment_seconds)):
        segment = output / f"segment_{index:03d}"
        segment.mkdir()
        if selected:
            reference = segment / "reference.png"
            extract_last_frame(selected[-1]["video"], reference)
        segment_duration = min(
            args.segment_seconds, duration - index * args.segment_seconds
        )
        result = generate_segment(
            args, index, segment, reference, segment_duration, ffmpeg
        )
        selected.append(result)
        (output / "manifest.json").write_text(
            json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    concat = output / "concat.txt"
    concat.write_text(
        "".join(
            "file '" + item["video"].replace("'", "'\\''") + "'\n" for item in selected
        ),
        encoding="utf-8",
    )
    run(
        [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat,
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            output / "result.mp4",
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy", type=Path)
    parser.add_argument("--reference-image", type=Path)
    parser.add_argument("--prompt")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    args = parser.parse_args()
    if all((args.proxy, args.reference_image, args.prompt, args.output)):
        assemble(args)
    else:
        parser.error("--proxy, --reference-image, --prompt and --output are required")


if __name__ == "__main__":
    main()

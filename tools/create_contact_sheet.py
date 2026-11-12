"""Create a timestamped contact sheet from a video."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
from PIL import Image, ImageDraw


def sampled_indices(frame_count: int, sample_count: int) -> list[int]:
    """Return evenly spaced, de-duplicated frame indices."""
    if frame_count < 1:
        raise ValueError("The video contains no frames.")
    if sample_count < 1:
        raise ValueError("sample_count must be positive.")
    if sample_count == 1:
        return [0]
    return sorted(
        {
            round(position * (frame_count - 1) / (sample_count - 1))
            for position in range(sample_count)
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--columns", type=int, default=4)
    args = parser.parse_args()

    frames = list(iio.imiter(args.video, plugin="FFMPEG"))
    indices = sampled_indices(len(frames), args.count)
    tile_width, tile_height = 224, 128
    caption_height, margin = 24, 8
    rows = (len(indices) + args.columns - 1) // args.columns
    canvas = Image.new(
        "RGB",
        (
            args.columns * (tile_width + margin) + margin,
            rows * (tile_height + caption_height + margin) + margin,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)

    for position, frame_index in enumerate(indices):
        row, column = divmod(position, args.columns)
        x = margin + column * (tile_width + margin)
        y = margin + row * (tile_height + caption_height + margin)
        frame = Image.fromarray(frames[frame_index]).convert("RGB")
        canvas.paste(frame.resize((tile_width, tile_height)), (x, y))
        draw.text(
            (x, y + tile_height + 4),
            f"frame {frame_index} - {frame_index / args.fps:.2f}s",
            fill="black",
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)


if __name__ == "__main__":
    main()

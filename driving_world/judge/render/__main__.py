"""Command line entry point for render calibration and scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .render import (
    Canonical,
    RenderChecker,
    build_references,
    load_references,
    write_references,
    write_report,
)

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def _videos(directory: Path, limit: int | None) -> list[Path]:
    paths = sorted(
        path
        for path in directory.rglob("*")
        if path.suffix.lower() in VIDEO_SUFFIXES and path.is_file()
    )
    if not paths:
        raise SystemExit(f"no videos found under {directory}")
    return paths[:limit] if limit else paths


def main(argv: list[str] | None = None) -> int:
    """Run either the calibration build or a single-video check."""
    parser = argparse.ArgumentParser(prog="python -m driving_world.judge.render")
    commands = parser.add_subparsers(dest="command", required=True)

    calibrate = commands.add_parser("calibrate", help="build reference distributions")
    calibrate.add_argument("--videos", type=Path, required=True)
    calibrate.add_argument("--out", type=Path, required=True)
    calibrate.add_argument("--limit", type=int, default=None)
    calibrate.add_argument("--fps", type=float, default=10.0)
    calibrate.add_argument("--height", type=int, default=480)

    check = commands.add_parser("check", help="score one video")
    check.add_argument("--video", type=Path, required=True)
    check.add_argument("--references", type=Path, required=True)
    check.add_argument("--out", type=Path, default=None)
    check.add_argument("--fps", type=float, default=10.0)
    check.add_argument("--height", type=int, default=480)

    args = parser.parse_args(argv)
    if args.command == "calibrate":
        videos = _videos(args.videos, args.limit)
        canonical = Canonical(fps=args.fps, height=args.height)
        references = build_references(videos, canonical=canonical)
        write_references(args.out, references)
        print(
            json.dumps(
                {
                    "videos": len(videos),
                    "canonical": {"fps": canonical.fps, "height": canonical.height},
                    "axes": {
                        axis: {"probe": reference.probe, "samples": len(reference.samples)}
                        for axis, reference in references.items()
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    checker = RenderChecker(
        load_references(args.references),
        canonical=Canonical(fps=args.fps, height=args.height),
    )
    report = checker.check(args.video)
    if args.out is not None:
        write_report(args.out, report)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

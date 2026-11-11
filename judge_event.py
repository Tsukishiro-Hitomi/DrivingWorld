"""Run the baseline VLM object checker on a rendered H3 video against its proxy."""

import argparse
import json
from pathlib import Path

from driving_world.config import VLM_WEIGHTS
from driving_world.contracts.events import validate_event_spec
from driving_world.judge.object import ObjectChecker, write_result
from driving_world.judge.qwen import Qwen25VL


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path, help="rendered H3 video")
    parser.add_argument("--proxy-video", required=True, type=Path, help="proxy reference video")
    parser.add_argument("--event", required=True, type=Path, help="EventSpec JSON file")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-path", type=Path, default=VLM_WEIGHTS)
    args = parser.parse_args()
    if not args.video.is_file() or not args.proxy_video.is_file() or not args.event.is_file():
        raise FileNotFoundError("--video, --proxy-video and --event must exist")
    event = validate_event_spec(json.loads(args.event.read_text(encoding="utf-8")))
    result = ObjectChecker(Qwen25VL(str(args.model_path))).judge(
        args.video, args.proxy_video, event
    )
    write_result(args.output, result)
    print(f"Object checker report: {args.output} ({result.status})")


if __name__ == "__main__":
    main()

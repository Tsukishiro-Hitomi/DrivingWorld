"""Prepare controlled, unlabeled Event Judge candidates and optional proxy videos."""

import argparse
from pathlib import Path

from driving_world.datasets.pilot import prepare_collection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", type=int, default=2)
    parser.add_argument("--render-proxy", action="store_true")
    args = parser.parse_args()
    candidates = prepare_collection(args.output, render_proxy=args.render_proxy, contexts=args.contexts)
    print(f"Prepared {len(candidates)} candidates in {args.output}")


if __name__ == "__main__":
    main()

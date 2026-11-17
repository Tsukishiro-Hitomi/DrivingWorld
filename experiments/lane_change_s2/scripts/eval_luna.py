"""GPT-6-Luna multimodal judge for PREVENTION S2.

Runs one authenticated Codex CLI request per 8-frame contact sheet and appends
results as they complete so the batch is safely resumable.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import os
import subprocess
import tempfile
import threading
from pathlib import Path


from _paths import REPO, S2, repo_script, scratch_path  # noqa: E402
RESULT = S2 / "results/luna.jsonl"
SHEETS = scratch_path("luna_sheets")
SCHEMA = scratch_path("luna_schema.json")
MODEL = "gpt-6-luna"

PROMPT = """You are the multimodal judge for a lane-change benchmark.

The attached image contains exactly 8 frames sampled uniformly from one 8-second
dashcam clip, laid out left-to-right and then top-to-bottom in temporal order.
Each panel is numbered and timestamped. The camera is mounted on the ego car.
One target vehicle has a thick RED bounding box; other vehicles have thin green
boxes. Judge only the red-boxed target.

Estimate the target vehicle's lateral position relative to fixed road lane
markings in panel 1 and panel 8. Express the net signed lateral shift (panel 8
minus panel 1) in lane widths: negative means left and positive means right.
Compensate for ego/camera motion by using the road markings and scene, not pixel
position alone. A lane change is YES only when the target crosses a lane boundary
and the magnitude of its net shift is at least 0.5 lane widths. Do not count road
curvature, camera yaw, target size/depth change, or within-lane weaving.

Return a JSON object matching the supplied schema. Set pred=1 for YES and pred=0
for NO. Set shift to your signed lane-width estimate. In raw, give a very short
answer containing FIRST, LAST, SHIFT, and ANSWER (yes/no), without extra analysis.
"""

SCHEMA_VALUE = {
    "type": "object",
    "properties": {
        "pred": {"type": "integer", "enum": [0, 1]},
        "shift": {"type": ["number", "null"]},
        "raw": {"type": "string"},
    },
    "required": ["pred", "shift", "raw"],
    "additionalProperties": False,
}

_write_lock = threading.Lock()


def load_grid_image():
    spec = importlib.util.spec_from_file_location("eval_s2", repo_script("eval_s2.py"))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.grid_image


def make_sheet(clip_id: str, grid_image) -> Path:
    SHEETS.mkdir(parents=True, exist_ok=True)
    target = SHEETS / f"{clip_id}.jpg"
    if not target.exists():
        image = grid_image(S2 / "clips" / f"{clip_id}.mp4")
        image.save(target, format="JPEG", quality=90, optimize=True)
    return target


def parse_response(text: str) -> dict:
    text = text.strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start : end + 1])
    pred = value.get("pred")
    if pred not in (0, 1):
        raise ValueError(f"invalid pred: {pred!r}")
    shift = value.get("shift")
    if shift is not None:
        shift = float(shift)
    return {"pred": int(pred), "shift": shift, "raw": str(value.get("raw", ""))[:500]}


def judge_one(clip_id: str, sheet: Path) -> dict:
    errors = []
    for attempt in (1, 2):
        with tempfile.NamedTemporaryFile(prefix=f"luna_{clip_id}_", suffix=".json", delete=False) as fh:
            output = Path(fh.name)
        command = [
            "codex", "exec", "--ephemeral", "--ignore-rules",
            "--model", MODEL, "--sandbox", "read-only", "--cd", str(REPO),
            "--image", str(sheet), "--output-schema", str(SCHEMA),
            "--output-last-message", str(output), PROMPT,
        ]
        try:
            completed = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=360,
                check=False,
            )
            response = output.read_text(encoding="utf-8") if output.exists() else ""
            if completed.returncode != 0:
                raise RuntimeError(f"exit {completed.returncode}: {completed.stdout[-1200:]}")
            parsed = parse_response(response)
            return {"clip_id": clip_id, **parsed}
        except Exception as error:  # one retry, then null per task contract
            errors.append(f"attempt {attempt}: {type(error).__name__}: {error}")
        finally:
            output.unlink(missing_ok=True)
    return {"clip_id": clip_id, "pred": None, "shift": None, "raw": " | ".join(errors)[-500:]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    SCHEMA.write_text(json.dumps(SCHEMA_VALUE, indent=2), encoding="utf-8")
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if RESULT.exists():
        for line in RESULT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["clip_id"])
    samples = json.loads((S2 / "samples.json").read_text(encoding="utf-8"))
    pending = [row for row in samples if row["clip_id"] not in done]
    if args.limit is not None:
        pending = pending[: args.limit]
    print(f"[luna] model={MODEL} resume={len(done)} pending={len(pending)} workers={args.workers}", flush=True)

    grid_image = load_grid_image()
    jobs = []
    for sample in pending:
        clip_id = sample["clip_id"]
        try:
            jobs.append((clip_id, make_sheet(clip_id, grid_image)))
        except Exception as error:
            row = {"clip_id": clip_id, "pred": None, "shift": None,
                   "raw": f"sheet error: {type(error).__name__}: {error}"[:500]}
            with RESULT.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[luna] {clip_id} -> null (sheet error)", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(judge_one, clip_id, sheet): clip_id for clip_id, sheet in jobs}
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            with _write_lock, RESULT.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            print(f"[luna] {row['clip_id']} -> {row['pred']} shift={row['shift']}", flush=True)

    print(f"[luna] wrote/resumed {len(done) + len(pending)} records", flush=True)


if __name__ == "__main__":
    main()

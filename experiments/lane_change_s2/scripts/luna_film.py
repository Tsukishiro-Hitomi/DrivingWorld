#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from _paths import REPO, RESULTS, S2, scratch_path  # noqa: E402
SAMPLES = S2 / "samples.json"
SHEETS = S2 / "sheets_film"
OUTPUT = RESULTS / "film_luna.jsonl"
TMP_ROOT = scratch_path("film_luna_work")
MODEL = "gpt-6-luna"
PROMPT = """This single image is a 2x4 grid of 8 crops sampled uniformly from one 8-second dashcam clip, in temporal order left-to-right then top-to-bottom. Each crop is centred on the single target vehicle, which is outlined with a thick RED box; thinner green boxes are other vehicles. Each panel is stamped with its panel number and time. The camera is mounted on the ego car.
Judge ONLY the red-boxed vehicle: did it move out of its lane into an adjacent lane (cross a lane boundary) during the clip?
Work through these steps and do not skip any.
STEP 1: In panel 1, identify which lane the red vehicle is in, then state where its body sits inside that lane in lane widths measured from the lane marking on its left (0 = touching the left line, 1 = touching the right line). Quote the number.
STEP 2: Do the same measurement in panel 8.
STEP 3: Give the signed difference, last minus first, in lane widths: negative means it moved left, positive means it moved right.
STEP 4: Answer yes if the magnitude of that difference is at least 0.5 lane widths OR you clearly see the red vehicle change lanes between panel 1 and panel 8; otherwise no.
Reply exactly in this form, one per line:
FIRST: <number> lane widths
LAST: <number> lane widths
SHIFT: <signed number>
ANSWER: yes
or
ANSWER: no"""


def free_gb():
    return shutil.disk_usage("/").free / (1024 ** 3)


def parse_answer(raw):
    answers = re.findall(r"(?im)^\s*ANSWER:\s*(yes|no)\s*$", raw)
    shifts = re.findall(r"(?im)^\s*SHIFT:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*$", raw)
    pred = None if not answers else int(answers[-1].lower() == "yes")
    shift = None if not shifts else float(shifts[-1])
    return pred, shift


def run_one(record):
    clip_id = record["clip_id"]
    image = SHEETS / f"{clip_id}.png"
    errors = []
    for attempt in (1, 2):
        work = Path(tempfile.mkdtemp(prefix=f"{clip_id}_", dir=TMP_ROOT))
        env = os.environ.copy()
        env["TMPDIR"] = str(work)
        try:
            proc = subprocess.run(
                ["codex", "exec", "--model", MODEL, "--image", str(image),
                 "--ephemeral", "--color", "never", PROMPT],
                cwd=REPO, env=env, stdin=subprocess.DEVNULL, text=True,
                capture_output=True, timeout=300,
            )
            raw = proc.stdout.strip()
            pred, shift = parse_answer(raw)
            if proc.returncode == 0 and pred is not None and shift is not None:
                return {"clip_id": clip_id, "pred": pred, "shift": shift, "raw": raw}
            errors.append(
                f"attempt {attempt}: exit={proc.returncode}; "
                f"parse_pred={pred}; parse_shift={shift}; stderr={proc.stderr.strip()[-500:]}"
            )
        except subprocess.TimeoutExpired:
            errors.append(f"attempt {attempt}: timeout after 300s")
        except Exception as exc:
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
        finally:
            shutil.rmtree(work, ignore_errors=True)
    return {"clip_id": clip_id, "pred": None, "shift": None, "raw": " | ".join(errors)}


def load_samples():
    data = json.loads(SAMPLES.read_text())
    if isinstance(data, dict):
        for key in ("samples", "records", "data"):
            if key in data:
                return data[key]
    return data


def report(samples, rows):
    meta = {r["clip_id"]: r for r in samples}
    by = defaultdict(list)
    for row in rows:
        if row["clip_id"] in meta:
            by[meta[row["clip_id"]]["stratum"]].append(row)
    print("REPORT")
    for stratum in ("cut_in", "cut_out", "lateral", "negative_clean", "negative_hard"):
        rs = by[stratum]
        denom = sum(1 for x in samples if x["stratum"] == stratum)
        judged = [x for x in rs if x["pred"] is not None]
        numer = sum(x["pred"] == 1 for x in judged)
        metric = "recall" if not stratum.startswith("negative") else "FPR"
        print(f"{stratum}: {metric} full={numer}/{denom}={numer/denom:.6f}; "
              f"judged={numer}/{len(judged)}={numer/len(judged):.6f}" if judged else
              f"{stratum}: {metric} full=0/{denom}=0.000000; judged=0/0=NA")
    pos = [r for r in rows if r["clip_id"] in meta and meta[r["clip_id"]]["label"] == 1]
    neg = [r for r in rows if r["clip_id"] in meta and meta[r["clip_id"]]["label"] == 0]
    posj = [r for r in pos if r["pred"] is not None]
    negj = [r for r in neg if r["pred"] is not None]
    hits = sum(r["pred"] == 1 for r in posj)
    fps = sum(r["pred"] == 1 for r in negj)
    print(f"overall recall full={hits}/76={hits/76:.6f}; judged={hits}/{len(posj)}="
          f"{hits/len(posj):.6f}" if posj else "overall recall full=0/76=0.000000; judged=0/0=NA")
    print(f"overall FPR full={fps}/145={fps/145:.6f}; judged={fps}/{len(negj)}="
          f"{fps/len(negj):.6f}" if negj else "overall FPR full=0/145=0.000000; judged=0/0=NA")
    small = sum(r["shift"] is not None and 0.0 <= abs(r["shift"]) <= 0.2 for r in pos)
    print(f"positive_abs_shift_0.0_to_0.2={small}/76")
    print(f"nulls={sum(r['pred'] is None for r in rows)}")
    print("pred_counts", dict(Counter(r["pred"] for r in rows)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    TMP_ROOT.mkdir(exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    samples = load_samples()
    existing = []
    if OUTPUT.exists():
        for line in OUTPUT.read_text().splitlines():
            if line.strip():
                existing.append(json.loads(line))
    done = {r["clip_id"] for r in existing}
    pending = [r for r in samples if r["clip_id"] not in done]
    if args.limit is not None:
        pending = pending[:args.limit]
    print(f"model={MODEL} samples={len(samples)} existing={len(done)} pending={len(pending)} free_gb={free_gb():.2f}", flush=True)
    for start in range(0, len(pending), args.batch_size):
        if free_gb() < 5.0:
            raise RuntimeError(f"root free space below 5 GB before batch: {free_gb():.2f} GB")
        batch = pending[start:start + args.batch_size]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(run_one, batch))
        with OUTPUT.open("a", encoding="utf-8") as fh:
            for row in results:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
        for child in list(TMP_ROOT.iterdir()):
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        print(f"completed={len(done) + min(start + len(batch), len(pending))}/{len(done) + len(pending)} "
              f"batch_nulls={sum(r['pred'] is None for r in results)} free_gb={free_gb():.2f}", flush=True)
    rows = []
    if OUTPUT.exists():
        rows = [json.loads(x) for x in OUTPUT.read_text().splitlines() if x.strip()]
    report(samples, rows)
    shutil.rmtree(TMP_ROOT, ignore_errors=True)


if __name__ == "__main__":
    main()

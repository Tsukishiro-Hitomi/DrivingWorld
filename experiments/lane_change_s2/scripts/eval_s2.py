"""Evaluate lane-change judges on the PREVENTION S2 set.

Routes:
  a1     geometric criterion on the ground-truth subject track
  vlm    Qwen3-VL-8B on a single 2x4 contact sheet of the clip
  report  per-stratum recall / false-positive rate

Unlike the Waymo smoke set, the positives here are human annotations that no
detector produced, so recall is finally a real number.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from _paths import REPO, S2, QWEN3_VL_8B, legacy_dir  # noqa: E402
sys.path.insert(0, str(REPO))

from driving_world.benchmark.prevention import (  # noqa: E402
    FPS,
    _sample_at,
    load_trajectories,
)

ANNOT = legacy_dir("prevention_annot")
RESULTS = S2 / "results"
TOLERANCE = 8
DEPTH_BAND = 20.0
LATERAL_RANGE = (5.0, 100.0)
MIN_OTHER = 2
NET_RANGE = (1.8, 6.5)


def load_samples():
    return json.loads((S2 / "samples.json").read_text(encoding="utf-8"))


def tracks_for(drive):
    cache = tracks_for.__dict__.setdefault("_cache", {})
    if drive not in cache:
        cache[drive] = load_trajectories(ANNOT / drive / "trajectories.txt")
    return cache[drive]


def window_series(track, start, count):
    xs = np.full(count, np.nan)
    depths = np.full(count, np.nan)
    for offset in range(count):
        sampled = _sample_at(track, start + offset, TOLERANCE)
        if sampled is not None:
            xs[offset] = sampled[0]
            depths[offset] = sampled[1]
    return xs, depths


def smooth(values, window=5):
    known = np.isfinite(values)
    if known.sum() < 2:
        return values
    filled = np.interp(np.arange(len(values)), np.flatnonzero(known), values[known])
    pad = window // 2
    padded = np.concatenate([np.full(pad, filled[0]), filled, np.full(pad, filled[-1])])
    return np.convolve(padded, np.ones(window) / window, mode="valid")[: len(values)]


def criterion(track, tracks, start, count, corrected):
    xs, depths = window_series(track, start, count)
    if np.isfinite(xs).sum() < count // 2:
        return None
    speed = np.gradient(smooth(xs)) * FPS
    if corrected:
        subject_depth = float(np.nanmedian(depths))
        rows = []
        for vehicle, other in tracks.items():
            if other is track:
                continue
            oxs, odepths = window_series(other, start, count)
            if np.isfinite(oxs).sum() < count // 2:
                continue
            if not np.isfinite(odepths).any():
                continue
            if abs(float(np.nanmedian(odepths)) - subject_depth) > DEPTH_BAND:
                continue
            median_x = float(np.nanmedian(oxs))
            if not LATERAL_RANGE[0] < median_x < LATERAL_RANGE[1] and abs(median_x) > 8:
                continue
            rows.append(np.gradient(smooth(oxs)) * FPS)
        if len(rows) >= MIN_OTHER:
            rate = np.nanmedian(np.vstack(rows), axis=0)
            speed = speed - np.nan_to_num(rate)
    displacement = np.concatenate([[0.0], np.cumsum(speed)[:-1]]) / FPS
    net = float(np.median(displacement[-10:]) - np.median(displacement[:10]))
    return net


def run_a1():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample in load_samples():
        tracks = tracks_for(sample["drive"])
        track = tracks.get(sample["vehicle"])
        if track is None:
            rows.append({"clip_id": sample["clip_id"], "pred": 0, "pred_nc": 0,
                         "net": 0.0, "net_nc": 0.0, "missing": True})
            continue
        net = criterion(track, tracks, sample["clip_start"], sample["clip_frames"], True)
        net_nc = criterion(track, tracks, sample["clip_start"], sample["clip_frames"], False)
        rows.append(
            {
                "clip_id": sample["clip_id"],
                "pred": int(net is not None and NET_RANGE[0] <= abs(net) <= NET_RANGE[1]),
                "pred_nc": int(net_nc is not None and NET_RANGE[0] <= abs(net_nc) <= NET_RANGE[1]),
                "net": round(net, 3) if net is not None else None,
                "net_nc": round(net_nc, 3) if net_nc is not None else None,
            }
        )
        print(f"[a1] {sample['clip_id']} net={rows[-1]['net']} -> {rows[-1]['pred']}", flush=True)
    with (RESULTS / "a1.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"[a1] wrote {len(rows)}")


def grid_image(path, count=8, cols=4, panel_w=800):
    """Aspect-preserving contact sheet; smaller panels keep the VLM pass affordable."""
    import imageio.v3 as iio
    from PIL import Image, ImageDraw

    decoded = list(iio.imiter(path, plugin="FFMPEG"))
    if not decoded:
        raise RuntimeError(f"no frames in {path}")
    height, width = decoded[0].shape[:2]
    panel_h = max(1, round(panel_w * height / width))
    picks = sorted({round(i * (len(decoded) - 1) / (count - 1)) for i in range(count)})
    rows = (len(picks) + cols - 1) // cols
    sheet = Image.new("RGB", (panel_w * cols, panel_h * rows), (0, 0, 0))
    draw = ImageDraw.Draw(sheet)
    for slot, index in enumerate(picks):
        panel = Image.fromarray(decoded[index]).resize((panel_w, panel_h), Image.LANCZOS)
        x, y = (slot % cols) * panel_w, (slot // cols) * panel_h
        sheet.paste(panel, (x, y))
        draw.rectangle([x, y, x + 150, y + 22], fill=(0, 0, 0))
        draw.text((x + 6, y + 5), f"{slot + 1}/{len(picks)}  t={index / FPS:.1f}s", fill=(255, 255, 0))
    return sheet


VLM_PROMPT = (
    "This single image holds 8 frames sampled uniformly from one 8-second dashcam clip, "
    "laid out left-to-right then top-to-bottom in temporal order. Each panel is stamped "
    "with its panel number and time. The camera is mounted on the ego car.\n"
    "One vehicle has a thick RED box; thinner green boxes are other vehicles. Judge only "
    "the red vehicle.\n"
    "Work through these steps and do not skip any.\n"
    "STEP 1: In panel 1, state where the red vehicle's body sits inside its lane, measured "
    "in lane widths from the lane marking on its left (0 = touching the left line, 1 = "
    "touching the right line). Quote the number.\n"
    "STEP 2: Do the same measurement in panel 8.\n"
    "STEP 3: Give the signed difference, last minus first, in lane widths: negative means "
    "it moved left, positive means it moved right.\n"
    "STEP 4: Say yes if the magnitude of that difference is at least 0.5 lane widths "
    "(the vehicle crossed a lane boundary), otherwise no.\n"
    "Reply exactly in this form, one per line:\n"
    "FIRST: <number> lane widths\nLAST: <number> lane widths\nSHIFT: <signed number>\n"
    "ANSWER: yes\nor\nANSWER: no"
)


def run_vlm():
    device = "cuda:0"
    model_path = str(QWEN3_VL_8B)
    import re

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = (
        Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=device, local_files_only=True
        )
        .eval()
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    done = set()
    partial = RESULTS / "vlm.jsonl"
    if partial.exists():
        for line in partial.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows.append(row)
                done.add(row["clip_id"])
        print(f"[vlm] resuming with {len(done)} finished", flush=True)
    for sample in load_samples():
        if sample["clip_id"] in done:
            continue
        clip = S2 / "clips" / f"{sample['clip_id']}.mp4"
        if not clip.exists():
            continue
        try:
            image = grid_image(clip)
        except Exception as error:  # noqa: BLE001 - one unreadable clip must not stop the run
            print(f"[vlm] skip {sample['clip_id']}: {type(error).__name__}: {error}", flush=True)
            continue
        text = processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": VLM_PROMPT},
                    ],
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=320, do_sample=False)
        raw = processor.batch_decode(
            output[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )[0].strip()
        answer = re.search(r"ANSWER:\s*(yes|no)", raw, re.I)
        shift = re.search(r"SHIFT:\s*([-+]?\d+(?:\.\d+)?)", raw, re.I)
        rows.append(
            {
                "clip_id": sample["clip_id"],
                "pred": None if answer is None else int(answer.group(1).lower() == "yes"),
                "shift": None if shift is None else float(shift.group(1)),
            }
        )
        print(f"[vlm] {sample['clip_id']} -> {rows[-1]['pred']} shift={rows[-1]['shift']}", flush=True)
        if len(rows) % 10 == 0:
            with (RESULTS / "vlm.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
    with (RESULTS / "vlm.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"[vlm] wrote {len(rows)}")


def report():
    samples = {s["clip_id"]: s for s in load_samples()}
    for path in sorted(RESULTS.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows:
            continue
        for key in ("pred", "pred_nc"):
            if key not in rows[0]:
                continue
            by_stratum: dict[str, list[tuple[int, int]]] = {}
            for row in rows:
                sample = samples.get(row["clip_id"])
                if sample is None or row.get(key) is None:
                    continue
                by_stratum.setdefault(sample["stratum"], []).append(
                    (sample["label"], int(row[key]))
                )
            print(f"\n=== {path.stem} [{key}] ===")
            positives = sum(t for pairs in by_stratum.values() for t, _ in pairs)
            for stratum in ("cut_in", "cut_out", "lateral", "negative_clean", "negative_hard"):
                pairs = by_stratum.get(stratum)
                if not pairs:
                    continue
                guessed = sum(g for _, g in pairs)
                if all(t == 1 for t, _ in pairs):
                    print(f"  {stratum:<16} recall {guessed}/{len(pairs)} = {guessed / len(pairs):.2f}")
                else:
                    print(f"  {stratum:<16} FPR    {guessed}/{len(pairs)} = {guessed / len(pairs):.2f}")
            hits = sum(g for pairs in by_stratum.values() for t, g in pairs if t == 1)
            print(f"  {'ALL positives':<16} {hits}/{positives} = {hits / positives:.2f}")
            negs = [(t, g) for pairs in by_stratum.values() for t, g in pairs if t == 0]
            if negs:
                fp = sum(g for _, g in negs)
                print(f"  {'ALL negatives':<16} {fp}/{len(negs)} = {fp / len(negs):.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("a1", "vlm", "report"))
    args = parser.parse_args()
    {"a1": run_a1, "vlm": run_vlm, "report": report}[args.command]()


if __name__ == "__main__":
    main()

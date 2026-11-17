"""Evaluate Qwen3-VL-8B on the box-normalised filmstrips."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from _paths import REPO, S2, QWEN3_VL_8B  # noqa: E402
FILM = S2 / "sheets_film1b"
RESULTS = S2 / "results"

FILM_PROMPT = (
    "You are shown one image: a 2x4 grid of 8 panels (1920x480). Read the panels left to "
    "right, top row first: panels 1-4 are the top row, 5-8 the bottom row. Every panel is the "
    "same dashcam view at a different moment of one 8-second clip (0.0 s to 7.9 s, about "
    "1.1 s apart), cropped tightly around ONE target vehicle, so the vehicle keeps the same "
    "size in every panel.\n"
    "Ignore the small red rectangle: it is a dataset marker, not a cue.\n"
    "On every panel a cyan horizontal line marks the target vehicle's tyre-contact row, and "
    "the cyan ticks along that line are spaced exactly ONE CAR WIDTH apart. The target "
    "vehicle is about 1.8 m wide; a motorway lane is 3.5 m, which is about 2 car widths. Use "
    "those ticks as the ruler.\n"
    "Your task: decide whether the target vehicle completed a lane change during this clip, "
    "that is, whether it moved from one lane into the neighbouring lane so that a lane "
    "marking passed from one side of the vehicle to the other.\n"
    "Use the sign convention: positive means the vehicle sits to the RIGHT of its lane "
    "centre, negative means to the LEFT.\n"
    "Answer in exactly four lines, nothing else, no explanation:\n"
    "FIRST: <signed number> car widths\n"
    "LAST: <signed number> car widths\n"
    "SHIFT: <signed number> car widths\n"
    "ANSWER: yes or no"
)


def load_samples():
    return json.loads((S2 / "samples.json").read_text(encoding="utf-8"))


def _parse(raw: str):
    answer = re.search(r"ANSWER:\s*(yes|no)", raw, re.I)
    shift = re.search(r"SHIFT:\s*([-+]?\d+(?:\.\d+)?)", raw, re.I)
    return (
        None if answer is None else int(answer.group(1).lower() == "yes"),
        None if shift is None else float(shift.group(1)),
    )


def run_qwen(limit: int | None, tag: str):
    device = "cuda:1"
    model_path = str(QWEN3_VL_8B)
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device, local_files_only=True
    ).eval()
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{tag}.jsonl"
    rows, done = [], set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows.append(row)
                done.add(row["clip_id"])
        print(f"[qwen] resume {len(done)}", flush=True)
    samples = load_samples()
    if limit:
        samples = samples[:limit]
    for i, sample in enumerate(samples):
        cid = sample["clip_id"]
        if cid in done:
            continue
        path = FILM / f"{cid}.png"
        if not path.exists():
            print(f"[qwen] missing {cid}", flush=True)
            continue
        image = Image.open(path).convert("RGB")
        text = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image", "image": image},
                                          {"type": "text", "text": FILM_PROMPT}]}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = processor(text=[text], images=[image], padding=True,
                           return_tensors="pt").to(device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=320, do_sample=False)
        raw = processor.batch_decode(output[:, inputs.input_ids.shape[1]:],
                                     skip_special_tokens=True)[0].strip()
        pred, shift = _parse(raw)
        rows.append({"clip_id": cid, "pred": pred, "shift": shift, "raw": raw[:300]})
        print(f"[qwen] {cid} -> {pred} shift={shift}", flush=True)
        if len(rows) % 10 == 0:
            out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                           encoding="utf-8")
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                   encoding="utf-8")
    print(f"[qwen] wrote {len(rows)}", flush=True)


def report(tag: str):
    samples = {s["clip_id"]: s for s in load_samples()}
    path = RESULTS / f"{tag}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    by = {}
    allp = []
    for row in rows:
        s = samples.get(row["clip_id"])
        if s is None or row.get("pred") is None:
            continue
        by.setdefault(s["stratum"], []).append((s["label"], row["pred"]))
        allp.append((s["label"], row["pred"]))
    for st in ["cut_in", "cut_out", "lateral", "negative_clean", "negative_hard"]:
        pairs = by.get(st, [])
        if not pairs:
            continue
        g = sum(g for _, g in pairs)
        kind = "recall" if all(t == 1 for t, _ in pairs) else "FPR   "
        print(f"  {st:<16} {kind} {g}/{len(pairs)} = {g/len(pairs):.4f}")
    hits = sum(g for t, g in allp if t == 1)
    pos = sum(1 for t, _ in allp if t == 1)
    fps = sum(g for t, g in allp if t == 0)
    neg = sum(1 for t, _ in allp if t == 0)
    print(f"  ALL positives recall {hits}/{pos} = {hits/max(pos,1):.4f}")
    print(f"  ALL negatives FPR    {fps}/{neg} = {fps/max(neg,1):.4f}")
    zeros = sum(1 for r in rows if r.get("shift") == 0.0)
    print(f"  SHIFT exactly 0: {zeros}/{len(rows)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("qwen", "report"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="film1b_qwen")
    a = ap.parse_args()
    if a.command == "qwen":
        run_qwen(a.limit, a.tag)
    else:
        report(a.tag)


if __name__ == "__main__":
    main()

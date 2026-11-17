"""Evaluate subject-crop filmstrip judges on the PREVENTION S2 set."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from _paths import REPO, S2, QWEN3_VL_8B  # noqa: E402
FILM = S2 / "sheets_film"
RESULTS = S2 / "results"

FILM_PROMPT = (
    "This single image is a 2x4 grid of 8 crops sampled uniformly from one 8-second dashcam "
    "clip, in temporal order left-to-right then top-to-bottom. Each crop is centred on the "
    "single target vehicle, which is outlined with a thick RED box; thinner green boxes are "
    "other vehicles. Each panel is stamped with its panel number and time. The camera is "
    "mounted on the ego car.\n"
    "Judge ONLY the red-boxed vehicle: did it move out of its lane into an adjacent lane "
    "(cross a lane boundary) during the clip?\n"
    "Work through these steps and do not skip any.\n"
    "STEP 1: In panel 1, identify which lane the red vehicle is in, then state where its body "
    "sits inside that lane in lane widths measured from the lane marking on its left "
    "(0 = touching the left line, 1 = touching the right line). Quote the number.\n"
    "STEP 2: Do the same measurement in panel 8.\n"
    "STEP 3: Give the signed difference, last minus first, in lane widths: negative means it "
    "moved left, positive means it moved right.\n"
    "STEP 4: Answer yes if the magnitude of that difference is at least 0.5 lane widths OR you "
    "clearly see the red vehicle change lanes between panel 1 and panel 8; otherwise no.\n"
    "Reply exactly in this form, one per line:\n"
    "FIRST: <number> lane widths\nLAST: <number> lane widths\nSHIFT: <signed number>\n"
    "ANSWER: yes\nor\nANSWER: no"
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
    device = "cuda:0"
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
                row = json.loads(line); rows.append(row); done.add(row["clip_id"])
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
            print(f"[qwen] missing film {cid}", flush=True); continue
        image = Image.open(path).convert("RGB")
        text = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": FILM_PROMPT}]}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=320, do_sample=False)
        raw = processor.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
        pred, shift = _parse(raw)
        rows.append({"clip_id": cid, "pred": pred, "shift": shift, "raw": raw[:300]})
        print(f"[qwen] {cid} -> {pred} shift={shift}", flush=True)
        if len(rows) % 10 == 0:
            out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"[qwen] wrote {len(rows)}", flush=True)


def report(tag: str):
    samples = {s["clip_id"]: s for s in load_samples()}
    path = RESULTS / f"{tag}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by = {}
    for row in rows:
        s = samples.get(row["clip_id"])
        if s is None or row.get("pred") is None:
            continue
        by.setdefault(s["stratum"], []).append((s["label"], row["pred"]))
    order = ["cut_in", "cut_out", "lateral", "negative_clean", "negative_hard"]
    for st in order:
        pairs = by.get(st, [])
        if not pairs:
            continue
        if all(t == 1 for t, _ in pairs):
            g = sum(g for _, g in pairs)
            print(f"  {st:<16} recall {g}/{len(pairs)} = {g/len(pairs):.4f}")
        else:
            g = sum(g for _, g in pairs)
            print(f"  {st:<16} FPR    {g}/{len(pairs)} = {g/len(pairs):.4f}")
    hits = sum(g for pairs in by.values() for t, g in pairs if t == 1)
    pos = sum(1 for pairs in by.values() for t, _ in pairs if t == 1)
    fps = sum(g for pairs in by.values() for t, g in pairs if t == 0)
    neg = sum(1 for pairs in by.values() for t, _ in pairs if t == 0)
    print(f"  ALL positives recall {hits}/{pos} = {hits/pos:.4f}")
    print(f"  ALL negatives FPR    {fps}/{neg} = {fps/neg:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("qwen", "report"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="film_qwen")
    a = ap.parse_args()
    if a.command == "qwen":
        run_qwen(a.limit, a.tag)
    else:
        report(a.tag)


if __name__ == "__main__":
    main()

"""Build 1b filmstrips: box-normalised crops with a car-width ruler.

The subject's detector box drives the crop, so the car occupies a constant
fraction of every panel and the road markings slide relative to the car -- the
lane change becomes visible as motion, not as a guess about absolute scale.
A ruler is drawn along the subject's contact row with ticks one car width apart,
which is exactly the scale the question needs.

Outputs
  experiments/lane_change_s2/sheets_film1b/<clip_id>.png
  experiments/lane_change_s2/results/film1b_boxes.jsonl
"""
import argparse
import json
from pathlib import Path

import numpy as np
import imageio.v3 as iio
from PIL import Image, ImageDraw

from _paths import REPO, S2  # noqa: E402
OUT = S2 / "sheets_film1b"
BOXES_OUT = S2 / "results/film1b_boxes.jsonl"
TEXT_H = 60
PANEL = (480, 240)
COUNT = 8
CROP_K = 3.0
BOX_BOTTOM_FRAC = 0.74
CLASS_WIDTH = {2: 1.80, 5: 2.55, 7: 2.50}


def pick_subject(frames):
    out = []
    for rec in frames:
        red = rec.get("red")
        dets = rec.get("dets") or []
        if red is None or not dets:
            out.append(None)
            continue
        cands = []
        for cls, conf, x1, y1, x2, y2 in dets:
            if not (x1 <= red["cx"] <= x2 and
                    y1 - 0.35 * (y2 - y1) <= red["cy"] <= y2 + 0.35 * (y2 - y1)):
                continue
            w, h = x2 - x1, y2 - y1
            if w < 24 or h < 16 or not (0.6 <= w / h <= 3.6):
                continue
            cands.append((conf, cls, x1, y1, x2, y2, w, h))
        if not cands:
            out.append(None)
            continue
        cands.sort(key=lambda c: c[0], reverse=True)
        c = cands[0]
        out.append({"cls": c[1], "u": (c[2] + c[4]) / 2.0, "w": c[6], "h": c[7],
                    "x1": c[2], "y1": c[3], "x2": c[4], "y2": c[5]})
    return out


def smooth_boxes(subj, min_frac=0.5):
    """Median-smooth the box width and interpolate short dropouts."""
    n = len(subj)
    idx = [i for i, b in enumerate(subj) if b is not None]
    if len(idx) < max(8, min_frac * n):
        return None
    fill = []
    for i in range(n):
        if subj[i] is not None:
            fill.append(subj[i])
            continue
        j = min(idx, key=lambda j: abs(j - i))
        fill.append(dict(subj[j]))
    ws = np.array([b["w"] for b in fill], dtype=float)
    us = np.array([b["u"] for b in fill], dtype=float)
    k = 5
    pad = k // 2
    ws_p = np.pad(ws, pad, mode="edge")
    us_p = np.pad(us, pad, mode="edge")
    ws_s = np.array([np.median(ws_p[i:i + k]) for i in range(n)])
    us_s = np.array([np.median(us_p[i:i + k]) for i in range(n)])
    for i, b in enumerate(fill):
        b["w_s"] = float(ws_s[i])
        b["u_s"] = float(us_s[i])
    return fill


def build(cid, s, frames_meta, frames_img):
    n = len(frames_img)
    subj = pick_subject(frames_meta)
    boxes = smooth_boxes(subj)
    if boxes is None:
        return None
    H, W = frames_img[0].shape[:2]
    picks = sorted({round(i * (n - 1) / (COUNT - 1)) for i in range(COUNT)})
    sheet = Image.new("RGB", (PANEL[0] * 4, PANEL[1] * 2), (0, 0, 0))
    draw = ImageDraw.Draw(sheet)
    meta = []
    for slot, idx in enumerate(picks):
        b = boxes[idx]
        Wc = int(np.clip(round(CROP_K * b["w_s"]), 120, W))
        Hc = int(round(Wc * PANEL[1] / PANEL[0]))
        X1 = int(np.clip(round(b["u_s"] - Wc / 2), 0, W - Wc))
        Y1 = int(np.clip(round(b["y2"] - BOX_BOTTOM_FRAC * Hc), 0, H - Hc))
        f = frames_img[idx].copy()
        f[:TEXT_H, :] = 0
        crop = f[Y1:Y1 + Hc, X1:X1 + Wc]
        px = (slot % 4) * PANEL[0]
        py = (slot // 4) * PANEL[1]
        sheet.paste(Image.fromarray(crop).resize(PANEL, Image.LANCZOS), (px, py))
        sx = PANEL[0] / Wc
        sy = PANEL[1] / Hc
        cxp = px + (b["u_s"] - X1) * sx
        yc = py + (b["y2"] - Y1) * sy
        draw.line([(px, yc), (px + PANEL[0], yc)], fill=(0, 255, 255), width=1)
        for k in (-2, -1, 0, 1, 2):
            x = cxp + k * b["w_s"] * sx
            if px <= x <= px + PANEL[0]:
                draw.line([(x, yc - 7), (x, yc + 7)], fill=(0, 255, 255), width=2)
        draw.rectangle([px, py, px + 172, py + 20], fill=(0, 0, 0))
        draw.text((px + 4, py + 4), f"{slot+1}/{COUNT} {idx/10:.1f}s", fill=(255, 255, 0))
        meta.append({"frame": idx, "u": round(b["u_s"], 1), "w": round(b["w_s"], 1),
                     "cls": b["cls"], "window": [X1, Y1, Wc, Hc]})
    sheet.save(OUT / f"{cid}.png")
    return {"clip_id": cid, "label": s["label"], "stratum": s.get("stratum"),
            "bw": round(float(np.median([b["w_s"] for b in boxes])), 1),
            "panels": meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    samples = json.loads((S2 / "samples.json").read_text(encoding="utf-8"))
    dets = {}
    for line in (S2 / "results/s2_dets.jsonl").open():
        if line.strip():
            r = json.loads(line)
            dets[r["clip_id"]] = r.get("frames") or []
    done = {p.stem for p in OUT.glob("*.png")}
    todo = [s for s in samples if s["clip_id"] not in done and s["clip_id"] in dets]
    if args.nshards > 1:
        todo = [s for k, s in enumerate(todo) if k % args.nshards == args.shard]
    if args.limit:
        todo = todo[:args.limit]
    print(f"todo {len(todo)}", flush=True)

    skipped = []
    boxes_out = Path(str(BOXES_OUT).replace(".jsonl", f".p{args.shard}.jsonl")) if args.nshards > 1 else BOXES_OUT
    with boxes_out.open("a") as fh:
        for i, s in enumerate(todo):
            cid = s["clip_id"]
            try:
                imgs = [np.asarray(f) for f in
                        iio.imiter(S2 / "clips" / f"{cid}.mp4", plugin="FFMPEG")]
                meta = build(cid, s, dets[cid], imgs)
                if meta is None:
                    skipped.append(cid)
                    print(f"[{i+1}/{len(todo)}] SKIP {cid} (subject box coverage)", flush=True)
                    continue
                fh.write(json.dumps(meta) + "\n")
                fh.flush()
                print(f"[{i+1}/{len(todo)}] {cid} bw={meta['bw']}", flush=True)
            except Exception as e:
                skipped.append(cid)
                print(f"[{i+1}/{len(todo)}] FAIL {cid}: {type(e).__name__}: {e}", flush=True)
    print("DONE skipped=%d" % len(skipped), flush=True)


if __name__ == "__main__":
    main()

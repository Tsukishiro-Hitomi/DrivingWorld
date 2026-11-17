"""Metric depth from DA3 on S2, sampled where the judge needs it.

Per sampled frame this records, in original 1920x600 pixel coordinates:
  * the depth statistics inside the subject's box (the vehicle's own depth),
  * the road depth on both sides of the vehicle along its contact row
    (the surface the lane markings sit on),
  * the per-image scale factor DA3 reports, so the metric form can be checked.

Depth from a depth model is the only thing missing from the detector-based
geometry: with GT depth the same pipeline reaches 0.908 / 0.166, with depth
guessed from the box width it drops to 0.855 / 0.269.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import imageio.v3 as iio

from _paths import REPO, S2  # noqa: E402
OUT = S2 / "results/da3_depth.jsonl"
MODEL_DIR = REPO / "models/da3metric-large"


def pick_subject(rec):
    red = rec.get("red")
    dets = rec.get("dets") or []
    if red is None or not dets:
        return None
    cands = []
    for cls, conf, x1, y1, x2, y2 in dets:
        if not (x1 <= red["cx"] <= x2 and
                y1 - 0.35 * (y2 - y1) <= red["cy"] <= y2 + 0.35 * (y2 - y1)):
            continue
        w, h = x2 - x1, y2 - y1
        if w < 24 or h < 16 or not (0.6 <= w / h <= 3.6):
            continue
        cands.append((conf, x1, y1, x2, y2))
    if not cands:
        return None
    cands.sort(key=lambda c: c[0], reverse=True)
    _, x1, y1, x2, y2 = cands[0]
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "u": (x1 + x2) / 2.0, "w": x2 - x1, "h": y2 - y1}


def region_stats(z, c, x1, y1, x2, y2):
    h, w = z.shape
    sx, sy = w / c["W_orig"], h / c["H_orig"]
    a = int(np.clip(round(x1 * sx), 0, w - 1))
    b = int(np.clip(round(x2 * sx), 0, w - 1))
    p = int(np.clip(round(y1 * sy), 0, h - 1))
    q = int(np.clip(round(y2 * sy), 0, h - 1))
    if b <= a or q <= p:
        return None
    patch = z[p:q + 1, a:b + 1]
    patch = patch[np.isfinite(patch)]
    if patch.size < 12:
        return None
    return {"med": float(np.median(patch)),
            "p25": float(np.percentile(patch, 25)),
            "p75": float(np.percentile(patch, 75)),
            "n": int(patch.size)}


def road_depth(z, c, box):
    """Depth of the road surface beside the car, along its contact row."""
    h, w = z.shape
    sx, sy = w / c["W_orig"], h / c["H_orig"]
    y = int(np.clip(round((box["y2"] - 1) * sy), 0, h - 1))
    out = {}
    for side, sgn in (("left", -1), ("right", 1)):
        a = box["u"] + sgn * 0.75 * box["w"]
        b = box["u"] + sgn * 1.60 * box["w"]
        lo, hi = min(a, b), max(a, b)
        a2 = int(np.clip(round(lo * sx), 0, w - 1))
        b2 = int(np.clip(round(hi * sx), 0, w - 1))
        if b2 - a2 < 2:
            out[side] = None
            continue
        seg = z[max(0, y - 1):y + 2, a2:b2 + 1]
        seg = seg[np.isfinite(seg)]
        out[side] = float(np.median(seg)) if seg.size >= 6 else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    import torch
    from depth_anything_3.api import DepthAnything3

    device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda")
    model = DepthAnything3.from_pretrained(str(MODEL_DIR)).to(device=device)
    model.eval()

    samples = json.loads((S2 / "samples.json").read_text(encoding="utf-8"))
    dets = {}
    for line in (S2 / "results/s2_dets.jsonl").open():
        if line.strip():
            r = json.loads(line)
            if r.get("frames"):
                dets[r["clip_id"]] = r["frames"]

    todo = [s for s in samples if s["clip_id"] in dets]
    if args.limit:
        todo = todo[:args.limit]
    if args.nshards > 1:
        todo = [s for k, s in enumerate(todo) if k % args.nshards == args.shard]
    print(f"todo {len(todo)}", flush=True)

    if args.selftest:
        s = todo[0]
        frames = [np.asarray(f) for f in
                  iio.imiter(S2 / "clips" / f"{s['clip_id']}.mp4", plugin="FFMPEG")]
        pred = model.inference([frames[10]], process_res=504)
        print("depth", pred.depth.shape, "isfinite", np.isfinite(pred.depth).mean())
        print("is_metric", pred.is_metric, "scale_factor", pred.scale_factor)
        print("intrinsics", None if pred.intrinsics is None else pred.intrinsics.shape,
              pred.intrinsics[0] if pred.intrinsics is not None else None)
        print("depth stats", float(np.nanmin(pred.depth)), float(np.nanmedian(pred.depth)),
              float(np.nanmax(pred.depth)))
        box = pick_subject(dets[s["clip_id"]][10])
        c = {"W_orig": frames[10].shape[1], "H_orig": frames[10].shape[0]}
        print("box", box)
        if box:
            print("box depth", region_stats(pred.depth[0], c, box["x1"], box["y1"],
                                            box["x2"], box["y2"]))
            print("road depth", road_depth(pred.depth[0], c, box))
        return

    out_path = S2 / (f"results/da3_depth.p{args.shard}.jsonl" if args.nshards > 1
                     else "results/da3_depth.jsonl")
    samples_by_id = {s["clip_id"]: s for s in samples}
    with out_path.open("a") as fh:
        for i, s in enumerate(todo):
            cid = s["clip_id"]
            try:
                frames = [np.asarray(f) for f in
                          iio.imiter(S2 / "clips" / f"{cid}.mp4", plugin="FFMPEG")]
                H, W = frames[0].shape[:2]
                c = {"W_orig": W, "H_orig": H}
                recs = []
                for idx in range(0, len(frames), args.stride):
                    meta = pick_subject(dets[cid][idx])
                    if meta is None:
                        recs.append({"i": idx, "box": None})
                        continue
                    pred = model.inference([frames[idx]], process_res=504)
                    z = pred.depth[0].astype(np.float32)
                    recs.append({
                        "i": idx,
                        "box": {k: round(float(v), 1) for k, v in meta.items()},
                        "depth": region_stats(z, c, meta["x1"], meta["y1"],
                                              meta["x2"], meta["y2"]),
                        "road": road_depth(z, c, meta),
                        "scale": None if pred.scale_factor is None else float(pred.scale_factor),
                        "is_metric": (int(pred.is_metric)
                                      if isinstance(pred.is_metric, (int, float, bool))
                                      else str(pred.is_metric)[:40]),
                    })
                fh.write(json.dumps({"clip_id": cid, "frames": recs}) + "\n")
                fh.flush()
                ds = [r["depth"]["med"] for r in recs if r.get("depth")]
                print(f"[{i+1}/{len(todo)}] {cid} n={len(recs)} "
                      f"z_med={np.median(ds):.1f}" if ds else
                      f"[{i+1}/{len(todo)}] {cid} n={len(recs)} no depth", flush=True)
            except Exception as e:
                print(f"[{i+1}/{len(todo)}] FAIL {cid}: {type(e).__name__}: {e}", flush=True)
                fh.write(json.dumps({"clip_id": cid, "frames": [], "error": str(e)}) + "\n")
                fh.flush()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

"""Detect vehicles on every S2 clip and record the annotation box alongside.

Writes one shard file per process, ``results/s2_dets.p<i>.jsonl``:
  {"clip_id": ..., "frames": [{"dets": [[cls, conf, x1, y1, x2, y2], ...],
                               "red": {"cx","cy","w","h"} | None}, ...]}

The red marker baked into the source video only says *which* vehicle is the
subject; every metric is computed from the detector box.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import imageio.v3 as iio
from ultralytics import YOLO

from _paths import REPO, S2, legacy_dir  # noqa: E402
TEXT_H = 60
VEHICLE_CLASSES = [2, 5, 7]


def _components(mask):
    """8-free BFS over a sparse boolean mask; returns bboxes plus pixel counts."""
    coords = set(zip(*np.nonzero(mask)))
    out = []
    while coords:
        st = coords.pop()
        stack = [st]
        n = 1
        ymin = ymax = st[0]
        xmin = xmax = st[1]
        while stack:
            y, x = stack.pop()
            if y < ymin: ymin = y
            if y > ymax: ymax = y
            if x < xmin: xmin = x
            if x > xmax: xmax = x
            for nb in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if nb in coords:
                    coords.remove(nb)
                    stack.append(nb)
                    n += 1
        out.append([xmin, ymin, xmax, ymax, n])
    return out


def _merge(comps, gap=6):
    """Union-find over component bboxes so an outline split by text rejoins."""
    parent = list(range(len(comps)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def near(a, b):
        return not (a[2] + gap < b[0] or b[2] + gap < a[0] or
                    a[3] + gap < b[1] or b[3] + gap < a[1])

    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            if near(comps[i], comps[j]):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    groups = {}
    for i, c in enumerate(comps):
        groups.setdefault(find(i), []).append(c)
    merged = []
    for g in groups.values():
        merged.append([min(c[0] for c in g), min(c[1] for c in g),
                       max(c[2] for c in g), max(c[3] for c in g),
                       sum(c[4] for c in g)])
    return merged


def red_box(frame):
    H = frame.shape[0]
    roi = frame[TEXT_H:int(0.62 * H)]
    arr = roi.astype(np.int16)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    m = (r > 120) & (r - g > 50) & (r - b > 50)
    if m.sum() < 40:
        return None
    best = None
    for x1, y1, x2, y2, n in _merge(_components(m)):
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        if w < 14 or h < 8 or not (1.0 <= w / h <= 9.0):
            continue
        if best is None or w * h > best[2] * best[3]:
            best = (x1, y1 + TEXT_H, w, h)
    if best is None:
        return None
    return {"cx": best[0] + best[2] / 2.0, "cy": best[1] + best[3] / 2.0,
            "w": float(best[2]), "h": float(best[3])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11m.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out_path = S2 / f"results/s2_dets.p{args.shard}.jsonl"

    samples = json.loads((S2 / "samples.json").read_text(encoding="utf-8"))
    done = set()
    if out_path.exists():
        for line in out_path.open():
            if line.strip():
                done.add(json.loads(line)["clip_id"])
    todo = [s for s in samples if s["clip_id"] not in done]
    todo = [s for k, s in enumerate(todo) if k % args.nshards == args.shard]
    if args.limit:
        todo = todo[:args.limit]
    print(f"shard {args.shard}/{args.nshards} todo {len(todo)}", flush=True)

    model = YOLO(args.model)
    with out_path.open("a") as fh:
        for i, s in enumerate(todo):
            cid = s["clip_id"]
            try:
                frames = [np.asarray(f) for f in
                          iio.imiter(S2 / "clips" / f"{cid}.mp4", plugin="FFMPEG")]
                results = model.predict(frames, imgsz=args.imgsz, conf=args.conf,
                                        classes=VEHICLE_CLASSES, verbose=False,
                                        device=0, batch=8)
                recs = []
                for fr, res in zip(frames, results):
                    boxes = res.boxes
                    dets = []
                    if boxes is not None and len(boxes):
                        xyxy = boxes.xyxy.cpu().numpy()
                        cls = boxes.cls.cpu().numpy()
                        conf = boxes.conf.cpu().numpy()
                        for k in range(len(xyxy)):
                            dets.append([int(cls[k]), round(float(conf[k]), 3)] +
                                        [round(float(v), 1) for v in xyxy[k]])
                    recs.append({"dets": dets, "red": red_box(fr)})
                fh.write(json.dumps({"clip_id": cid, "frames": recs}) + "\n")
                fh.flush()
                print(f"[{i+1}/{len(todo)}] {cid} red={sum(1 for r in recs if r['red'])}", flush=True)
            except Exception as e:
                print(f"[{i+1}/{len(todo)}] FAIL {cid}: {type(e).__name__}: {e}", flush=True)
                fh.write(json.dumps({"clip_id": cid, "frames": [], "error": str(e)}) + "\n")
                fh.flush()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

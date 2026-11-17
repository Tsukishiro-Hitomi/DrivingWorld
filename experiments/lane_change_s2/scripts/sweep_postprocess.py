"""Post-processing sweep on cached detections: can the FPR be trimmed cheaply?

The estimator is y = (u - cx0) * W / w with a per-class effective width, which
already reaches 0.855 / 0.269.  Before adding a depth model it is worth knowing
how much of that 0.269 is just noisy aggregation rather than depth error.
"""
import json
from pathlib import Path

import numpy as np

from _paths import REPO, S2, legacy_dir  # noqa: E402
ANNOT = legacy_dir("prevention_annot")
FITTED_WIDTH = {2: 2.35, 5: 2.45, 7: 2.90}
MOTION_MIN, MOTION_MAX = 1.8, 6.5
EDGE = 10


def load_tracks(path):
    t = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) != 10:
            continue
        t.setdefault(int(p[1]), {})[int(p[0])] = (float(p[4]), float(p[5]), float(p[6]))
    return t


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
        cands.append((conf, cls, (x1 + x2) / 2.0, w))
    if not cands:
        return None
    cands.sort(key=lambda c: c[0], reverse=True)
    c = cands[0]
    return {"cls": c[1], "u": c[2], "w": c[3], "conf": c[0]}


def smooth(v, k):
    if k <= 1:
        return np.asarray(v, float)
    pad = k // 2
    a = np.pad(np.asarray(v, float), pad, mode="edge")
    return np.array([np.median(a[i:i + k]) for i in range(len(v))])


def main():
    samples = {s["clip_id"]: s for s in
               json.loads((S2 / "samples.json").read_text(encoding="utf-8"))}
    labels = {json.loads(l)["clip_id"]: json.loads(l) for l in
              (S2 / "labels.jsonl").open() if l.strip()}
    dets = {}
    for line in (S2 / "results/s2_dets.jsonl").open():
        if line.strip():
            r = json.loads(line)
            if r.get("frames"):
                dets[r["clip_id"]] = r["frames"]
    tracks = {}
    for drive in sorted({s["drive"] for s in samples.values()}):
        p = ANNOT / drive / "trajectories.txt"
        if p.exists():
            tracks[drive] = load_tracks(p)

    clips = []
    for cid, s in samples.items():
        if cid not in dets:
            continue
        lab = labels[cid]
        start = int(s["clip_start"])
        track = tracks.get(s["drive"], {}).get(int(lab["track_id"]), {})
        boxes = [pick_subject(r) for r in dets[cid]]
        gts = []
        for i in range(len(boxes)):
            f = start + i
            gts.append(track.get(f))
        clips.append({"clip_id": cid, "label": lab["label"], "stratum": lab["stratum"],
                      "drive": s["drive"], "boxes": boxes, "gts": gts})

    cam = {}
    for drive in tracks:
        A, b = [], []
        for c in clips:
            if c["drive"] != drive:
                continue
            for box, gt in zip(c["boxes"], c["gts"]):
                if box is None or gt is None or gt[2] < 6:
                    continue
                A.append([1.0, gt[0] / gt[2]])
                b.append(box["u"])
        if len(A) < 50:
            continue
        sol, *_ = np.linalg.lstsq(np.asarray(A), np.asarray(b), rcond=None)
        cam[drive] = (float(sol[0]), float(sol[1]))

    def series(c, k=5, min_w=0):
        idx, us, ws = [], [], []
        for i, box in enumerate(c["boxes"]):
            if box is None or box["w"] < min_w:
                continue
            idx.append(i)
            us.append(box["u"])
            ws.append(box["w"])
        if len(idx) < 24:
            return None
        cx0, f = cam[c["drive"]]
        us = smooth(us, 5)
        ws = smooth(ws, k)
        ys = []
        for j, i in enumerate(idx):
            if ws[j] <= 0:
                continue
            W = FITTED_WIDTH.get(c["boxes"][i]["cls"], 1.8)
            ys.append((us[j] - cx0) * W / ws[j])
        return ys if len(ys) >= 20 else None

    def evaluate(name, agg, k=5, min_w=0):
        tp = fn = fp = tn = 0
        by = {}
        for c in clips:
            ys = series(c, k=k, min_w=min_w)
            pred = 0
            if ys is not None:
                d = agg(np.asarray(ys))
                pred = int(d is not None and MOTION_MIN <= abs(d) <= MOTION_MAX)
            s = c["stratum"]
            b = by.setdefault(s, [0, 0, 0, 0])
            if c["label"] == 1:
                b[1] += 1
                if pred: tp += 1; b[0] += 1
                else: fn += 1
            else:
                b[3] += 1
                if pred: fp += 1; b[2] += 1
                else: tn += 1
        print(f"{name:<40} recall={tp}/{tp+fn}={tp/max(tp+fn,1):.3f}  "
              f"fpr={fp}/{fp+tn}={fp/max(fp+tn,1):.3f}")
        return tp, tp + fn, fp, fp + tn

    def med_edge(y):
        return float(np.median(y[-EDGE:]) - np.median(y[:EDGE]))

    def linfit(y):
        n = len(y)
        t = np.arange(n, dtype=float)
        A = np.vstack([t, np.ones(n)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        return float(coef[0] * (n - 1))

    def trim_edge(y, frac=0.25):
        n = len(y)
        m = max(EDGE, int(frac * n))
        return float(np.median(y[-m:]) - np.median(y[:m]))

    def med_edge_monotone(y):
        return med_edge(y)

    print("aggregators (fitted W, median-5 smoothing on u):")
    evaluate("median first/last 10", med_edge)
    evaluate("least-squares slope x duration", linfit)
    evaluate("median first/last 25%", lambda y: trim_edge(y, 0.25))
    evaluate("median first/last 35%", lambda y: trim_edge(y, 0.35))
    print("\nreliability filters:")
    evaluate("min box width 40 px", med_edge, min_w=40)
    evaluate("min box width 60 px", med_edge, min_w=60)
    evaluate("min box width 80 px", med_edge, min_w=80)
    print("\nwidth smoothing:")
    for k in (1, 3, 5, 9, 15):
        evaluate(f"median smoothing k={k}", med_edge, k=k)


if __name__ == "__main__":
    main()

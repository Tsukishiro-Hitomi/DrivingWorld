"""Sweep detector-only lateral estimators on cached detections.

Everything here reuses `results/s2_dets.jsonl`; no video is decoded.  Variants
differ in how the metric scale is obtained (nominal car width, empirically
fitted effective width, or an oracle GT depth that bounds what detection error
can be blamed for), how the box series is smoothed, and which box dimension is
trusted.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from _paths import REPO, S2, legacy_dir  # noqa: E402
ANNOT = legacy_dir("prevention_annot")

NOMINAL_W = {2: 1.80, 5: 2.55, 7: 2.50}
MOTION_MIN = 1.8
MOTION_MAX = 6.5
EDGE = 10


def load_tracks(path):
    tracks = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) != 10:
            continue
        tracks.setdefault(int(p[1]), {})[int(p[0])] = (
            float(p[4]), float(p[5]), float(p[6]))
    return tracks


def sample_track(track, frame, tol=8):
    if frame in track:
        return track[frame]
    before = [f for f in track if f < frame]
    after = [f for f in track if f > frame]
    if not before or not after:
        return None
    lo, hi = max(before), min(after)
    if frame - lo > tol or hi - frame > tol:
        return None
    t = (frame - lo) / (hi - lo)
    return tuple(track[lo][k] * (1 - t) + track[hi][k] * t for k in range(3))


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
            cands.append((conf, cls, (x1 + x2) / 2.0, w, h))
        if not cands:
            out.append(None)
            continue
        cands.sort(key=lambda c: c[0], reverse=True)
        c = cands[0]
        out.append({"cls": c[1], "u": c[2], "w": c[3], "h": c[4]})
    return out


def interp(series):
    """Carry boxes across short dropouts; drop leading/trailing gaps."""
    n = len(series)
    idx = [i for i, b in enumerate(series) if b is not None]
    if not idx:
        return [None] * n
    lo, hi = idx[0], idx[-1]
    out = [None] * n
    for i in range(lo, hi + 1):
        if series[i] is not None:
            out[i] = dict(series[i])
            continue
        prev = max(j for j in idx if j < i)
        nxt = min(j for j in idx if j > i)
        t = (i - prev) / (nxt - prev)
        a, b = series[prev], series[nxt]
        out[i] = {"cls": a["cls"],
                  "u": a["u"] * (1 - t) + b["u"] * t,
                  "w": a["w"] * (1 - t) + b["w"] * t,
                  "h": a["h"] * (1 - t) + b["h"] * t}
    return out


def smooth(vals, k):
    if k <= 1:
        return np.asarray(vals, dtype=float)
    pad = k // 2
    v = np.pad(np.asarray(vals, dtype=float), pad, mode="edge")
    return np.array([np.median(v[i:i + k]) for i in range(len(vals))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", default=str(S2 / "results/s2_dets.jsonl"))
    args = ap.parse_args()

    samples = {s["clip_id"]: s for s in
               json.loads((S2 / "samples.json").read_text(encoding="utf-8"))}
    labels = {r["clip_id"]: r for r in
              (json.loads(l) for l in (S2 / "labels.jsonl").open() if l.strip())}
    dets = {}
    for line in Path(args.dets).open():
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
        boxes = interp(pick_subject(dets[cid]))
        gts = [sample_track(track, start + i) if track else None
               for i in range(len(boxes))]
        clips.append({"clip_id": cid, "label": lab["label"], "stratum": lab["stratum"],
                      "drive": s["drive"], "boxes": boxes, "gts": gts})

    # camera fit
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
        sol, *_ = np.linalg.lstsq(np.asarray(A), np.asarray(b), rcond=None)
        cam[drive] = (float(sol[0]), float(sol[1]))
        print(f"cam {drive}: cx0={sol[0]:.1f} f={sol[1]:.1f} n={len(A)}")

    # effective width per class: w * z_gt = f * W
    eff = {}
    for cls, nom in NOMINAL_W.items():
        vals = []
        for c in clips:
            f = cam[c["drive"]][1]
            for box, gt in zip(c["boxes"], c["gts"]):
                if box is None or gt is None or gt[2] < 8 or box["cls"] != cls:
                    continue
                vals.append(box["w"] * gt[2] / f)
        eff[cls] = float(np.median(vals)) if len(vals) > 30 else nom
        print(f"eff width cls{cls}: nominal {nom:.2f} -> fitted {eff[cls]:.2f} m (n={len(vals)})")

    def evaluate(name, wide, k, dim, oracle):
        tp = fn = fp = tn = 0
        by = {}
        for c in clips:
            lo = [i for i, b in enumerate(c["boxes"]) if b is not None]
            if len(lo) < 24:
                pred = 0
            else:
                idx = list(range(lo[0], lo[-1] + 1))
                bs = [c["boxes"][i] for i in idx]
                us = smooth([b["u"] for b in bs], k)
                ws = smooth([b[dim] for b in bs], k)
                cx0, f = cam[c["drive"]]
                if oracle:
                    z = np.array([(c["gts"][i][2]) if c["gts"][i] else np.nan
                                  for i in idx], dtype=float)
                else:
                    z = None
                ys = []
                for j, i in enumerate(idx):
                    if oracle:
                        if not np.isfinite(z[j]) or z[j] < 1:
                            continue
                        y = (us[j] - cx0) * z[j] / f
                    else:
                        if ws[j] <= 0:
                            continue
                        y = (us[j] - cx0) * wide(c["boxes"][i]["cls"]) / ws[j]
                    ys.append(y)
                if len(ys) < 20:
                    pred = 0
                else:
                    ys = np.asarray(ys)
                    d = float(np.median(ys[-EDGE:]) - np.median(ys[:EDGE]))
                    pred = int(MOTION_MIN <= abs(d) <= MOTION_MAX)
            s = c["stratum"]
            b = by.setdefault(s, [0, 0, 0, 0])
            if c["label"] == 1:
                b[1] += 1
                if pred:
                    tp += 1; b[0] += 1
                else:
                    fn += 1
            else:
                b[3] += 1
                if pred:
                    fp += 1; b[2] += 1
                else:
                    tn += 1
        print(f"{name:<34} recall={tp}/{tp+fn}={tp/max(tp+fn,1):.3f}  "
              f"fpr={fp}/{fp+tn}={fp/max(fp+tn,1):.3f}  " +
              " ".join(f"{s}:{by[s][0]}/{by[s][1]}" for s in
                       ("cut_in", "cut_out", "lateral") if s in by))
        return tp, tp + fn, fp, fp + tn

    print()
    evaluate("nominal W, k=1, w, box-depth", lambda c: NOMINAL_W.get(c, 1.8), 1, "w", False)
    evaluate("nominal W, k=5, w, box-depth", lambda c: NOMINAL_W.get(c, 1.8), 5, "w", False)
    evaluate("nominal W, k=9, w, box-depth", lambda c: NOMINAL_W.get(c, 1.8), 9, "w", False)
    evaluate("fitted W,  k=5, w, box-depth", lambda c: eff.get(c, 1.8), 5, "w", False)
    evaluate("fitted W,  k=9, w, box-depth", lambda c: eff.get(c, 1.8), 9, "w", False)
    evaluate("fitted W,  k=5, h, box-depth", lambda c: eff.get(c, 1.8), 5, "h", False)
    print()
    evaluate("ORACLE depth,  k=1", lambda c: 1.8, 1, "w", True)
    evaluate("ORACLE depth,  k=5", lambda c: 1.8, 5, "w", True)


if __name__ == "__main__":
    main()

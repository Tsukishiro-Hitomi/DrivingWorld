"""Refine the DA3-based judge: discrete lane index instead of a pure magnitude band.

With depth accurate to ~5-10% the reconstructed lateral position is good enough
that the lane-index discretisation can be applied directly, which on GT was the
best-performing form (0.961 / 0.152 for index-change + magnitude band).
"""
import json
from pathlib import Path

import numpy as np

from _paths import REPO, S2, legacy_dir  # noqa: E402
ANNOT = legacy_dir("prevention_annot")
LANE_W = 3.5
C = 982.0
EDGE = 10


def load_tracks(path):
    t = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) != 10:
            continue
        t.setdefault(int(p[1]), {})[int(p[0])] = (float(p[4]), float(p[5]), float(p[6]))
    return t


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
    w = (frame - lo) / (hi - lo)
    return tuple(track[lo][k] * (1 - w) + track[hi][k] * w for k in range(3))


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
    return {"cls": c[1], "u": c[2], "w": c[3]}


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
    da3 = {}
    for line in (S2 / "results/da3_depth.jsonl").open():
        if line.strip():
            r = json.loads(line)
            if r.get("frames"):
                da3[r["clip_id"]] = {f["i"]: f for f in r["frames"]}
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
        gts = [sample_track(track, start + i) if track else None
               for i in range(len(boxes))]
        clips.append({"clip_id": cid, "label": lab["label"], "stratum": lab["stratum"],
                      "drive": s["drive"], "boxes": boxes, "gts": gts,
                      "da3": da3.get(cid, {})})

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

    def series(c):
        cx0, f = cam[c["drive"]]
        out = []
        for i, box in enumerate(c["boxes"]):
            rec = c["da3"].get(i)
            if box is None or not rec or not rec.get("depth"):
                continue
            z = rec["depth"]["med"] * f / C
            out.append((box["u"] - cx0) * z / f)
        return np.asarray(out) if len(out) >= 20 else None

    def score(name, fn):
        tp = fn_ = fp = tn = 0
        by = {}
        for c in clips:
            y = series(c)
            pred = 0 if y is None else int(bool(fn(y)))
            s = c["stratum"]
            b = by.setdefault(s, [0, 0, 0, 0])
            if c["label"] == 1:
                b[1] += 1
                if pred: tp += 1; b[0] += 1
                else: fn_ += 1
            else:
                b[3] += 1
                if pred: fp += 1; b[2] += 1
                else: tn += 1
        print(f"{name:<42} recall={tp}/{tp+fn_}={tp/max(tp+fn_,1):.3f}  "
              f"fpr={fp}/{fp+tn}={fp/max(fp+tn,1):.3f}  " +
              " ".join(f"{s}:{by[s][0]}/{by[s][1]}" for s in
                       ("cut_in", "cut_out", "lateral") if s in by))

    def ends(y):
        m = max(EDGE, len(y) // 8)
        return float(np.median(y[:m])), float(np.median(y[-m:]))

    def band(y, lo=1.8, hi=6.5):
        a, b = ends(y)
        return lo <= abs(b - a) <= hi

    def discrete(y):
        a, b = ends(y)
        return round(b / LANE_W) != round(a / LANE_W)

    def both(y):
        a, b = ends(y)
        return (round(b / LANE_W) != round(a / LANE_W)) and 1.8 <= abs(b - a) <= 6.5

    def stable_both(y):
        m = max(EDGE, len(y) // 8)
        i0 = np.round(y[:m] / LANE_W)
        i1 = np.round(y[-m:] / LANE_W)
        if float((i0 == i0[-1]).mean()) < 0.8 or float((i1 == i1[-1]).mean()) < 0.8:
            return False
        return bool(i1[-1] != i0[-1]) and 1.8 <= abs(float(np.median(y[-m:]) - np.median(y[:m]))) <= 6.5

    print("DA3-corrected lateral series, different decision rules:")
    score("magnitude band [1.8, 6.5] m (current)", band)
    score("discrete lane index change", discrete)
    score("discrete + magnitude band", both)
    score("discrete + stable ends + band", stable_both)


if __name__ == "__main__":
    main()

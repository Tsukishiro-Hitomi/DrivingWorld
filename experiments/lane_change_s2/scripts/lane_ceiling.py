"""Ceiling of a discrete lane-index criterion, computed on GT tracks.

A1 measures continuous lateral displacement and thresholds its magnitude into
[1.8, 6.5] m.  A lane-crossing judge instead asks whether the subject's lane
*index* changed and stayed changed.  On GT we can evaluate that discretisation
directly, which tells us whether a lane-based judge is worth building at all:
if the discrete form does not beat A1's 0.957 / 0.187 here, a learned lane
detector would inherit the same ceiling.
"""
import json
from pathlib import Path

import numpy as np

from _paths import REPO, S2, legacy_dir  # noqa: E402
ANNOT = legacy_dir("prevention_annot")
LANE_W = 3.5


def load_tracks(path):
    t = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) != 10:
            continue
        t.setdefault(int(p[1]), {})[int(p[0])] = (float(p[4]), float(p[5]), float(p[6]))
    return t


def sample(track, frame, tol=8):
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


def main():
    samples = json.loads((S2 / "samples.json").read_text(encoding="utf-8"))
    labels = {json.loads(l)["clip_id"]: json.loads(l) for l in
              (S2 / "labels.jsonl").open() if l.strip()}
    tracks = {}
    for drive in sorted({s["drive"] for s in samples}):
        p = ANNOT / drive / "trajectories.txt"
        if p.exists():
            tracks[drive] = load_tracks(p)

    rows = []
    for s in samples:
        cid = s["clip_id"]
        lab = labels[cid]
        track = tracks.get(s["drive"], {}).get(int(lab["track_id"]), {})
        start = int(s["clip_start"])
        xs = []
        for i in range(int(s["clip_frames"])):
            g = sample(track, start + i) if track else None
            if g is not None:
                xs.append(g[0])
        rows.append({"clip_id": cid, "label": lab["label"], "stratum": lab["stratum"],
                     "x": xs, "n": len(xs)})

    def score(name, pred_fn):
        tp = fn = fp = tn = 0
        by = {}
        for r in rows:
            p = pred_fn(r)
            b = by.setdefault(r["stratum"], [0, 0, 0, 0])
            if r["label"] == 1:
                b[1] += 1
                if p: tp += 1; b[0] += 1
                else: fn += 1
            else:
                b[3] += 1
                if p: fp += 1; b[2] += 1
                else: tn += 1
        print(f"{name:<46} recall={tp}/{tp+fn}={tp/max(tp+fn,1):.3f}  "
              f"fpr={fp}/{fp+tn}={fp/max(fp+tn,1):.3f}  " +
              " ".join(f"{k}:{by[k][0]}/{by[k][1]}" for k in
                       ("cut_in", "cut_out", "lateral") if k in by))
        return tp, tp + fn, fp, fp + tn

    def lane_idx(r, edge):
        if r["n"] < 30:
            return None
        x = np.asarray(r["x"])
        m = max(6, len(x) // 8)
        a = float(np.median(x[:m]))
        b = float(np.median(x[-m:]))
        return a, b

    def continuous(r, lo=1.8, hi=6.5):
        v = lane_idx(r, 0)
        if v is None:
            return 0
        d = v[1] - v[0]
        return int(lo <= abs(d) <= hi)

    def discrete(r):
        v = lane_idx(r, 0)
        if v is None:
            return 0
        a, b = v
        return int(round(b / LANE_W) != round(a / LANE_W))

    def discrete_stable(r):
        """Lane index must be stable at both ends, not just different."""
        if r["n"] < 30:
            return 0
        x = np.asarray(r["x"])
        m = max(6, len(x) // 8)
        i0 = np.round(x[:m] / LANE_W)
        i1 = np.round(x[-m:] / LANE_W)
        if float((i0 == i0[-1]).mean()) < 0.8 or float((i1 == i1[-1]).mean()) < 0.8:
            return 0
        return int(i1[-1] != i0[-1])

    def discrete_plus(r):
        v = lane_idx(r, 0)
        if v is None:
            return 0
        a, b = v
        if round(b / LANE_W) == round(a / LANE_W):
            return 0
        return int(1.8 <= abs(b - a) <= 6.5)

    print("GT-only criteria (no perception at all):")
    score("continuous |dx| in [1.8, 6.5] m  (= A1 shape)", continuous)
    score("discrete lane index change", discrete)
    score("discrete + stable at both ends", discrete_stable)
    score("discrete index change + |dx| band", discrete_plus)

    # how often does the lane index change by more than one lane?
    multi = 0
    tot = 0
    for r in rows:
        if r["n"] < 30:
            continue
        v = lane_idx(r, 0)
        if v is None:
            continue
        a, b = v
        d = abs(round(b / LANE_W) - round(a / LANE_W))
        tot += 1
        if d >= 2:
            multi += 1
    print(f"\nclips whose lane index moves >= 2 lanes: {multi}/{tot}")


if __name__ == "__main__":
    main()

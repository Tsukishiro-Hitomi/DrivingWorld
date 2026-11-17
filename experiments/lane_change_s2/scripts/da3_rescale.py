"""Is DA3's depth only mis-scaled? Fit the documented focal rescaling and retest.

The model card says the network emits canonical depth and that multiplying by
the focal length gives metres.  The per-drive ratio between DA3 depth and GT
depth is nearly constant within a drive but differs across drives in inverse
proportion to the focal length, which is exactly that relationship.  So fit one
global constant `c` with `z_metric = z_da3 * f / c` and see whether the residual
error collapses and the judge recovers.
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

    # fit the canonical->metric constant: z_metric = z_da3 * f / c
    per_drive = {}
    for c in clips:
        f = cam[c["drive"]][1]
        vals = []
        for i, rec in c["da3"].items():
            d = rec.get("depth")
            gt = c["gts"][i] if i < len(c["gts"]) else None
            if not d or gt is None or gt[2] < 6:
                continue
            vals.append(d["med"] * f / gt[2])
        if vals:
            per_drive.setdefault(c["drive"], []).extend(vals)
    all_c = []
    print("focal constant c per drive (z_metric = z_da3 * f / c):")
    for d, v in per_drive.items():
        print(f"  {d}: c = {np.median(v):.0f}  (n={len(v)}, IQR "
              f"{np.percentile(v,25):.0f}..{np.percentile(v,75):.0f})")
        all_c.extend(v)
    C = float(np.median(all_c))
    print(f"  global c = {C:.0f}")

    # residual error after correction
    bands = {"10-20": [], "20-35": [], "35-50": [], "50+": []}
    for c in clips:
        f = cam[c["drive"]][1]
        for i, rec in c["da3"].items():
            d = rec.get("depth")
            gt = c["gts"][i] if i < len(c["gts"]) else None
            if not d or gt is None or gt[2] < 6:
                continue
            z = d["med"] * f / C
            k = ("10-20" if gt[2] < 20 else "20-35" if gt[2] < 35 else
                 "35-50" if gt[2] < 50 else "50+")
            bands[k].append((z - gt[2]) / gt[2])
    print("\nrelative depth error after the focal correction:")
    for k, v in bands.items():
        if v:
            print(f"  {k:>6} m: median {np.median(v):+.3f}  |err| median "
                  f"{np.median(np.abs(v)):.3f}  n={len(v)}")

    def score(name, get_z):
        tp = fn = fp = tn = 0
        by = {}
        for c in clips:
            ys = []
            for i, box in enumerate(c["boxes"]):
                if box is None:
                    continue
                z = get_z(c, i, box)
                if z is None:
                    continue
                cx0, f = cam[c["drive"]]
                ys.append((box["u"] - cx0) * z / f)
            pred = 0
            if len(ys) >= 20:
                d = float(np.median(ys[-EDGE:]) - np.median(ys[:EDGE]))
                pred = int(MOTION_MIN <= abs(d) <= MOTION_MAX)
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
              f"fpr={fp}/{fp+tn}={fp/max(fp+tn,1):.3f}  " +
              " ".join(f"{s}:{by[s][0]}/{by[s][1]}" for s in
                       ("cut_in", "cut_out", "lateral") if s in by))

    def z_raw(c, i, box):
        rec = c["da3"].get(i)
        return rec["depth"]["med"] if rec and rec.get("depth") else None

    def z_corr(c, i, box):
        z = z_raw(c, i, box)
        return None if z is None else z * cam[c["drive"]][1] / C

    def z_gt(c, i, box):
        gt = c["gts"][i]
        return gt[2] if gt else None

    def z_known(c, i, box):
        return cam[c["drive"]][1] * FITTED_WIDTH.get(box["cls"], 1.8) / box["w"] if box["w"] > 0 else None

    print()
    score("DA3 raw (canonical)", z_raw)
    score("DA3 x focal / c (metric)", z_corr)
    score("oracle GT depth", z_gt)
    score("known size (fitted W)", z_known)


if __name__ == "__main__":
    main()

"""Score a detector-only lateral estimator on S2.

For every clip the subject vehicle's box is taken from YOLO detections, guided
only by the red marker baked into the video (which identifies *which* car, not
how far it moved).  Metric lateral offset then follows from the car's known
width -- ``y = (u - cx0) * z / f`` with ``z = f * W / w`` -- so ``f`` cancels and
only the principal point and the class width are needed:

    y = (u - cx0) * W / w

``cx0`` and ``f`` are fitted once per drive against the GT tracks, which stands
in for the camera model the loop will get from the proxy.  No per-frame GT is
used for the measurement itself.
"""
import json
import argparse
from pathlib import Path

import numpy as np

from _paths import REPO, S2, legacy_dir  # noqa: E402
ANNOT = legacy_dir("prevention_annot")

#: class -> real width in metres (COCO: car 2, bus 5, truck 7)
CLASS_WIDTH = {2: 1.80, 5: 2.55, 7: 2.50}

MOTION_MIN = 1.8
MOTION_MAX = 6.5
EDGE = 10


def load_tracks(path):
    tracks = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) != 10:
            continue
        frame, vehicle = int(p[0]), int(p[1])
        tracks.setdefault(vehicle, {})[frame] = (float(p[4]), float(p[5]), float(p[6]))
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


def pick_subject(frames, min_frac=0.55):
    """Per frame, the detection containing the red marker with the best score."""
    out = []
    for rec in frames:
        red = rec.get("red")
        dets = rec.get("dets") or []
        if red is None or not dets:
            out.append(None)
            continue
        cands = []
        for cls, conf, x1, y1, x2, y2 in dets:
            if not (x1 <= red["cx"] <= x2 and y1 - 0.35 * (y2 - y1) <= red["cy"] <= y2 + 0.35 * (y2 - y1)):
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
        out.append({"cls": c[1], "conf": c[0], "u": (c[2] + c[4]) / 2.0,
                    "v": (c[3] + c[5]) / 2.0, "w": c[6], "h": c[7],
                    "x1": c[2], "y1": c[3], "x2": c[4], "y2": c[5]})
    n = sum(1 for o in out if o is not None)
    return out, n / max(len(out), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", default=str(S2 / "results/s2_dets.jsonl"))
    ap.add_argument("--out", default=str(S2 / "results/s2_dets_report.json"))
    args = ap.parse_args()

    samples = {s["clip_id"]: s for s in json.loads((S2 / "samples.json").read_text(encoding="utf-8"))}
    labels = {r["clip_id"]: r for r in
              (json.loads(l) for l in (S2 / "labels.jsonl").open() if l.strip())}
    dets = {}
    for line in Path(args.dets).open():
        if line.strip():
            r = json.loads(line)
            dets[r["clip_id"]] = r.get("frames") or []

    tracks_by_drive = {}
    for drive in sorted({s["drive"] for s in samples.values()}):
        p = ANNOT / drive / "trajectories.txt"
        if p.exists():
            tracks_by_drive[drive] = load_tracks(p)
    print("drives with tracks:", list(tracks_by_drive))

    # ---- per clip: subject boxes + GT pairing
    rows = []
    for cid, s in samples.items():
        if cid not in dets:
            continue
        lab = labels[cid]
        frames = dets[cid]
        subj, frac = pick_subject(frames)
        track = tracks_by_drive.get(s["drive"], {}).get(int(lab["track_id"]), {})
        start = int(s["clip_start"])
        pairs = []
        for i, box in enumerate(subj):
            if box is None:
                continue
            gt = sample_track(track, start + i) if track else None
            pairs.append((i, box, gt))
        rows.append({"clip_id": cid, "label": lab["label"], "stratum": lab["stratum"],
                     "drive": s["drive"], "cover": frac, "pairs": pairs,
                     "n_frames": len(frames), "span_x": (subj[0], subj[-1])})
    print(f"clips with detections: {len(rows)}")

    # ---- per drive: fit principal point and focal length on GT projections
    cam = {}
    for drive in tracks_by_drive:
        A, b = [], []
        for r in rows:
            if r["drive"] != drive:
                continue
            for i, box, gt in r["pairs"]:
                if gt is None or gt[2] < 6:
                    continue
                A.append([1.0, gt[0] / gt[2]])
                b.append(box["u"])
        if len(A) < 50:
            continue
        sol, *_ = np.linalg.lstsq(np.asarray(A), np.asarray(b), rcond=None)
        resid = np.asarray(A) @ sol - np.asarray(b)
        cam[drive] = {"cx0": float(sol[0]), "f": float(sol[1]),
                      "n": len(A), "resid_med": float(np.median(np.abs(resid)))}
        print(f"cam {drive}: cx0={sol[0]:.1f} f={sol[1]:.1f} n={len(A)} "
              f"|resid|_med={np.median(np.abs(resid)):.1f}px")

    # ---- estimator
    def series(r):
        c = cam.get(r["drive"])
        if not c:
            return None
        ys = []
        for i, box, gt in r["pairs"]:
            W = CLASS_WIDTH.get(box["cls"], 1.8)
            if box["w"] <= 0:
                continue
            ys.append((i, (box["u"] - c["cx0"]) * W / box["w"]))
        if len(ys) < 20:
            return None
        return ys

    def net(ys, lo=0, hi=None):
        arr = [(i, v) for i, v in ys if lo <= i <= (hi if hi is not None else 10 ** 9)]
        if len(arr) < 20:
            return None
        vals = [v for _, v in arr]
        return float(np.median(vals[-EDGE:]) - np.median(vals[:EDGE]))

    report = {"camera": cam, "clips": []}
    for r in rows:
        ys = series(r)
        rec = {"clip_id": r["clip_id"], "label": r["label"], "stratum": r["stratum"],
               "cover": round(r["cover"], 3)}
        if ys:
            d = net(ys)
            rec["net_m"] = None if d is None else round(d, 3)
            rec["pred"] = int(d is not None and MOTION_MIN <= abs(d) <= MOTION_MAX)
            rec["n_meas"] = len(ys)
            gts = [(i, gt) for i, box, gt in r["pairs"] if gt is not None]
            if gts:
                rec["gt_net_m"] = round(float(np.median([g[0] for _, g in gts[-EDGE:]]) -
                                              np.median([g[0] for _, g in gts[:EDGE]])), 3)
                errs = []
                for i, box, gt in r["pairs"]:
                    if gt is None or gt[2] < 6:
                        continue
                    W = CLASS_WIDTH.get(box["cls"], 1.8)
                    z_hat = cam[r["drive"]]["f"] * W / box["w"]
                    errs.append((gt[2], (z_hat - gt[2]) / gt[2]))
                if errs:
                    bands = {"10-20": [], "20-35": [], "35-50": [], "50+": []}
                    for z, e in errs:
                        k = ("10-20" if z < 20 else "20-35" if z < 35 else
                             "35-50" if z < 50 else "50+")
                        bands[k].append(e)
                    rec["depth_err"] = {k: (round(float(np.median(v)), 3) if v else None)
                                        for k, v in bands.items()}
        else:
            rec["pred"] = 0
            rec["net_m"] = None
        report["clips"].append(rec)

    Path(args.out).write_text(json.dumps(report, indent=1))

    # ---- metrics
    def metrics(sel):
        pos = [c for c in report["clips"] if c["label"] == 1 and sel(c)]
        neg = [c for c in report["clips"] if c["label"] == 0 and sel(c)]
        tp = sum(c.get("pred", 0) for c in pos)
        fp = sum(c.get("pred", 0) for c in neg)
        return tp, len(pos), fp, len(neg)

    print("\n%-16s %-14s %-14s" % ("stratum", "recall", "fpr"))
    strata = ["cut_in", "cut_out", "lateral", "negative_clean", "negative_hard"]
    for s in strata:
        tp, np_, fp, nn = metrics(lambda c, s=s: c["stratum"] == s)
        print("%-16s %-14s %-14s" % (s, f"{tp}/{np_}={tp/max(np_,1):.3f}",
                                     f"{fp}/{nn}={fp/max(nn,1):.3f}"))
    tp, np_, fp, nn = metrics(lambda c: True)
    print("%-16s %-14s %-14s" % ("ALL", f"{tp}/{np_}={tp/max(np_,1):.3f}",
                                 f"{fp}/{nn}={fp/max(nn,1):.3f}"))

    print("\ndepth error (median rel) by GT range:")
    agg = {}
    for c in report["clips"]:
        for k, v in (c.get("depth_err") or {}).items():
            if v is not None:
                agg.setdefault(k, []).append(v)
    for k in ["10-20", "20-35", "35-50", "50+"]:
        v = agg.get(k) or []
        print(f"  {k:>6} m: " + (f"{np.median(v):+.3f} (n={len(v)})" if v else "none"))

    cov = [c["cover"] for c in report["clips"]]
    print(f"\nsubject box coverage: median {np.median(cov):.2f}, "
          f"p10 {np.percentile(cov,10):.2f}, clips<0.5: {sum(1 for c in cov if c < 0.5)}")


if __name__ == "__main__":
    main()

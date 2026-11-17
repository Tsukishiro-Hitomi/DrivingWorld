"""Numeric probe: near-field line fitting under the subject, extrapolated up."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import imageio.v3 as iio

from _paths import S2, SCRIPTS  # noqa: E402
spec = importlib.util.spec_from_file_location(
    "lc", SCRIPTS / "lane_crossing.py"
)
lc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lc)

cid = "pos_record1drive1_10733"
dets = {}
for line in (S2 / "results/s2_dets.jsonl").open():
    if line.strip():
        r = json.loads(line)
        if r.get("frames"):
            dets[r["clip_id"]] = r["frames"]
frames = [np.asarray(f) for f in iio.imiter(S2 / "clips" / f"{cid}.mp4", plugin="FFMPEG")]
boxes = [lc.pick_subject(r) for r in dets[cid]]

H, W = frames[0].shape[:2]
print("H", H, "W", W)


def near_points(frame, b):
    lo = frame.min(axis=2).astype(np.int16)
    pts = []
    for y in range(int(b["y2"]) + 6, H - 2, 2):
        for x in lc.row_runs(lo[y]):
            if abs(x - b["u"]) > 8.0 * b["w"]:
                continue
            pts.append((float(y), float(x)))
    return pts


def fit(pts, tol=4.0, min_inliers=25, max_lines=5):
    if len(pts) < min_inliers:
        return []
    P = np.asarray(pts)
    lines = []
    used = np.zeros(len(P), bool)
    for _ in range(max_lines):
        Q = P[~used]
        if len(Q) < min_inliers:
            break
        best = None
        step = max(1, len(Q) // 220)
        idx = range(0, len(Q), step)
        for i in idx:
            for j in idx:
                if j <= i:
                    continue
                dy = Q[j, 0] - Q[i, 0]
                if abs(dy) < 60:
                    continue
                bb = (Q[j, 1] - Q[i, 1]) / dy
                if abs(bb) > 3.0:
                    continue
                a = Q[i, 1] - bb * Q[i, 0]
                inl = np.abs(Q[:, 1] - (a + bb * Q[:, 0])) < tol
                n = int(inl.sum())
                if n >= min_inliers and (best is None or n > best[0]):
                    best = (n, a, bb, inl)
        if best is None:
            break
        n, a, bb, inl = best
        lines.append((n, a, bb))
        idx_all = np.nonzero(~used)[0][inl]
        used[idx_all] = True
    return lines


for i in (0, 26, 53, 79):
    b = boxes[i]
    if b is None:
        print(i, "no box"); continue
    pts = near_points(frames[i], b)
    lines = fit(pts)
    xs = sorted(round(a + bb * b["y2"]) for _, a, bb in lines)
    print(f"f{i} u={b['u']:.0f} w={b['w']:.0f} y2={b['y2']:.0f} pts={len(pts)} "
          f"lines={len(lines)} xs_at_y2={xs} slope={[round(l[2],2) for l in lines]}")

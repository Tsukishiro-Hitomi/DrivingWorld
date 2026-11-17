"""Lane-crossing judge v2: fit lane lines in image space, extrapolate to contact row.

A single-row search fails whenever the contact row lands in a dash gap, and
widening the vertical band does not help because the gap is longitudinal too.
So the paint points are collected over a range of rows below and around the
subject, straight lines are fitted by RANSAC (straight road -> straight lines in
the image), and each line is extrapolated to the subject's contact row.  At that
row the markings and the vehicle share a depth, so dividing their pixel offset by
the vehicle's own box width is scale free -- one car width is one box width.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import imageio.v3 as iio
from PIL import Image, ImageDraw

from _paths import REPO, S2  # noqa: E402
OUT_DIR = S2 / "results"

TEXT_H = 60
CAR_MASK = 0.60
SEARCH = 8.0
ROW_UP = 46            # rows scanned above the contact row
ROW_STEP = 3
MIN_RUN, MAX_RUN = 2, 42
LANE_CW = 3.5 / 1.8
JUMP_LO, JUMP_HI = 1.15, 2.90


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
        cands.append((conf, (x1 + x2) / 2.0, w, y2))
    if not cands:
        return None
    cands.sort(key=lambda c: c[0], reverse=True)
    _, u, w, y2 = cands[0]
    return {"u": u, "w": w, "y2": y2}


def row_runs(lo_row):
    """Thin bright runs in one row of the min-channel image."""
    idx = np.nonzero(lo_row > 100)[0]
    if idx.size < 2:
        return []
    runs = []
    start = prev = int(idx[0])
    for x in idx[1:]:
        x = int(x)
        if x - prev <= 2:
            prev = x
            continue
        runs.append((start, prev))
        start = prev = x
    runs.append((start, prev))
    out = []
    for a, b in runs:
        width = b - a + 1
        if not (MIN_RUN <= width <= MAX_RUN):
            continue
        left = lo_row[max(0, a - 15):max(1, a - 3)]
        right = lo_row[b + 4:min(len(lo_row), b + 16)]
        flank = np.concatenate([left, right])
        if flank.size == 0:
            continue
        if lo_row[a:b + 1].mean() < flank.mean() + 16:
            continue
        out.append((a + b) / 2.0)
    return out


def paint_points(frame, y2, u, w):
    """Paint points in the near field below the subject.

    The rows right next to the subject are the worst place to look: the markings
    there are steep and often fall in a dash gap.  The near field two hundred
    rows below has large, well separated stripes, and because lane lines are
    straight in the image the fitted lines extrapolate back up to the subject.
    """
    H = frame.shape[:2][0]
    lo = frame.min(axis=2).astype(np.int16)
    pts = []
    for y in range(int(y2) + 6, H - 2, 2):
        for x in row_runs(lo[y]):
            if abs(x - u) > SEARCH * w:
                continue
            pts.append((float(y), float(x)))
    return pts


def fit_lines(pts, tol=4.0, min_inliers=25, max_lines=5):
    """Greedy RANSAC over (row, x) pairs."""
    if len(pts) < min_inliers:
        return []
    P = np.asarray(pts)
    lines = []
    used = np.zeros(len(P), dtype=bool)
    for _ in range(max_lines):
        Q = P[~used]
        if len(Q) < min_inliers:
            break
        best = None
        step = max(1, len(Q) // 220)
        cand = list(range(0, len(Q), step))
        for ii in cand:
            for jj in cand:
                if jj <= ii:
                    continue
                dy = Q[jj, 0] - Q[ii, 0]
                if abs(dy) < 60:
                    continue
                b = (Q[jj, 1] - Q[ii, 1]) / dy
                if abs(b) > 3.0:
                    continue
                a = Q[ii, 1] - b * Q[ii, 0]
                res = np.abs(Q[:, 1] - (a + b * Q[:, 0]))
                inl = res < tol
                n = int(inl.sum())
                if n >= min_inliers and (best is None or n > best[0]):
                    best = (n, a, b, inl)
        if best is None:
            break
        n, a, b, inl = best
        if any(abs(b - lb) < 0.05 and abs((a + b * P[:, 0].mean()) -
                                          (la + lb * P[:, 0].mean())) < 10
               for _, la, lb in lines):
            used[np.nonzero(~used)[0][inl]] = True
            continue
        lines.append((n, a, b))
        used[np.nonzero(~used)[0][inl]] = True
    return lines


def lane_position(frame, box):
    u, w, y2 = box["u"], box["w"], box["y2"]
    pts = paint_points(frame, y2, u, w)
    lines = fit_lines(pts)
    xs = sorted(a + b * y2 for _, a, b in lines)
    xs = [x for x in xs if abs(x - u) <= SEARCH * w]
    left = [x for x in xs if x < u - CAR_MASK * w]
    right = [x for x in xs if x > u + CAR_MASK * w]
    if left and right:
        centre = (left[-1] + right[0]) / 2.0
        return (u - centre) / w, "both"
    if left or right:
        c = left[-1] if left else right[0]
        return (u - c) / w, "left" if left else "right"
    return None, "none"


def evaluate_clip(boxes, frames):
    series, kinds = [], []
    for fr, b in zip(frames, boxes):
        if b is None:
            series.append(None)
            kinds.append("none")
            continue
        pos, kind = lane_position(fr, b)
        series.append(pos)
        kinds.append(kind)
    return series, kinds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--debug", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="lanecross")
    args = ap.parse_args()

    samples = json.loads((S2 / "samples.json").read_text(encoding="utf-8"))
    labels = {json.loads(l)["clip_id"]: json.loads(l) for l in
              (S2 / "labels.jsonl").open() if l.strip()}
    dets = {}
    for line in (S2 / "results/s2_dets.jsonl").open():
        if line.strip():
            r = json.loads(line)
            if r.get("frames"):
                dets[r["clip_id"]] = r["frames"]

    todo = [s for s in samples if s["clip_id"] in dets]
    if args.debug:
        todo = [s for s in samples if s["clip_id"] in args.debug.split(",")]
    else:
        if args.nshards > 1:
            todo = [s for k, s in enumerate(todo) if k % args.nshards == args.shard]
        if args.limit:
            todo = todo[:args.limit]
    print(f"todo {len(todo)}", flush=True)

    if args.debug:
        out_path = None
    else:
        out_path = OUT_DIR / (f"{args.tag}.p{args.shard}.jsonl" if args.nshards > 1
                              else f"{args.tag}.jsonl")
    rows = []
    for i, s in enumerate(todo):
        cid = s["clip_id"]
        try:
            frames = [np.asarray(f) for f in
                      iio.imiter(S2 / "clips" / f"{cid}.mp4", plugin="FFMPEG")]
            boxes = [pick_subject(r) for r in dets[cid]]
            series, kinds = evaluate_clip(boxes, frames)
            vals = [(j, v) for j, v in enumerate(series) if v is not None]
            jumps = []
            net = 0.0
            for (j0, v0), (j1, v1) in zip(vals, vals[1:]):
                d = v1 - v0
                if JUMP_LO <= abs(d) <= JUMP_HI:
                    jumps.append({"frame": j1, "delta": round(float(d), 3)})
                    net += float(d)
            cover = len(vals) / max(len(series), 1)
            both = sum(1 for k in kinds if k == "both") / max(len(kinds), 1)
            row = {"clip_id": cid, "label": labels[cid]["label"],
                   "stratum": labels[cid]["stratum"], "cover": round(cover, 3),
                   "both": round(both, 3), "n_jumps": len(jumps),
                   "net": round(net, 3), "lanes": round(net / LANE_CW, 2),
                   "jumps": jumps[:8]}
            rows.append(row)
            if args.debug:
                draw_debug(cid, frames, boxes, series, kinds)
            print(f"[{i+1}/{len(todo)}] {cid} cover={cover:.2f} both={both:.2f} "
                  f"jumps={len(jumps)} lanes={row['lanes']}", flush=True)
        except Exception as e:
            print(f"[{i+1}/{len(todo)}] FAIL {cid}: {type(e).__name__}: {e}", flush=True)

    if out_path is not None:
        with out_path.open("a") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print("DONE", len(rows), flush=True)

    if rows and not args.debug:
        for thr in (0.5, 0.75, 1.0):
            hit = sum(1 for r in rows if abs(r["lanes"]) >= thr)
            print(f"|lanes| >= {thr}: {hit}/{len(rows)}")


def draw_debug(cid, frames, boxes, series, kinds):
    picks = [0, len(frames) // 3, 2 * len(frames) // 3, len(frames) - 1]
    sheet = Image.new("RGB", (480 * 4, 300), (0, 0, 0))
    d = ImageDraw.Draw(sheet)
    for k, idx in enumerate(picks):
        b = boxes[idx]
        if b is None:
            continue
        W = int(3 * b["w"])
        H = int(W * 300 / 480)
        X1 = int(np.clip(b["u"] - W / 2, 0, frames[idx].shape[1] - W))
        Y1 = int(np.clip(b["y2"] - 0.74 * H, 0, frames[idx].shape[0] - H))
        f = frames[idx].copy()
        f[:TEXT_H, :] = 0
        im = Image.fromarray(f[Y1:Y1 + H, X1:X1 + W]).resize((480, 300))
        sheet.paste(im, (k * 480, 0))
        sx = 480 / W
        sy = 300 / H
        pts = paint_points(frames[idx], b["y2"], b["u"], b["w"])
        lines = fit_lines(pts)
        for n, a, bb in lines:
            x_at = a + bb * b["y2"]
            x = k * 480 + (x_at - X1) * sx
            y0 = k * 0
            if k * 480 - 20 <= x <= k * 480 + 500:
                d.line([(x, 0), (x, 300)], fill=(0, 255, 0), width=2)
        for (py, px) in pts:
            if py < Y1 or py > Y1 + H:
                continue
            d.point((k * 480 + (px - X1) * sx, (py - Y1) * sy), fill=(255, 255, 0))
        cx = k * 480 + (b["u"] - X1) * sx
        d.line([(cx, 0), (cx, 300)], fill=(255, 0, 0), width=1)
        yc = (b["y2"] - Y1) * sy
        d.line([(k * 480, yc), (k * 480 + 480, yc)], fill=(0, 200, 255), width=1)
        pos = series[idx]
        d.rectangle([k * 480, 0, k * 480 + 190, 18], fill=(0, 0, 0))
        d.text((k * 480 + 4, 3),
               f"f{idx} {kinds[idx]} pos={'None' if pos is None else round(pos, 2)}",
               fill=(255, 255, 0))
    sheet.save(OUT_DIR / f"lanecross_dbg_{cid}.png")
    print("wrote dbg", cid, flush=True)


if __name__ == "__main__":
    main()

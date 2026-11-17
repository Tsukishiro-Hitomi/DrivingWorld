"""Pilot: box-normalized (1b) crop + lane-marking ruler, with debug overlays.

For a handful of S2 clips this
  1. tracks the annotation box with size/position continuity,
  2. crops each sampled frame to CROP_K x box-width centred on the box,
  3. finds bright lane-marking runs at the subject's contact row,
  4. draws the detected markings back into the crop so the ruler can be eyeballed,
and prints the lane-relative position series in car-width units.
"""
import argparse
import json
import numpy as np
from pathlib import Path
import imageio.v3 as iio
from PIL import Image, ImageDraw

from _paths import REPO, S2  # noqa: E402
OUT = S2 / "probe_film1b"
TEXT_H = 60
PANEL = (480, 240)
COUNT = 8
CROP_K = 3.0
BOX_BOTTOM_FRAC = 0.70


def _dilate2(mask, k=2, times=2):
    out = mask.copy()
    for _ in range(times):
        acc = out.copy()
        for dy in range(-k, k + 1):
            for dx in range(-k, k + 1):
                acc |= np.roll(np.roll(out, dy, axis=0), dx, axis=1)
        out = acc
    return out


def red_components(frame):
    """Annotation boxes are hollow rectangles with red text inside, so the raw
    connected components are outline fragments. Dilate to merge, then take bboxes."""
    arr = frame.astype(np.int16)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    m = (r > 120) & (r - g > 50) & (r - b > 50)
    m[:TEXT_H, :] = False
    ys, xs = np.nonzero(m)
    if ys.size == 0:
        return []
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    sub = m[y0:y1 + 1, x0:x1 + 1]
    d = _dilate2(sub)
    coords = set(zip(*np.nonzero(d)))
    comps = []
    while coords:
        st = coords.pop()
        stack = [st]
        n = 1
        ymin = ymax = st[0]
        xmin = xmax = st[1]
        while stack:
            y, x = stack.pop()
            ymin = min(ymin, y); ymax = max(ymax, y)
            xmin = min(xmin, x); xmax = max(xmax, x)
            for nb in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if nb in coords:
                    coords.remove(nb)
                    stack.append(nb)
                    n += 1
        w = xmax - xmin + 1 - 4
        h = ymax - ymin + 1 - 4
        if n >= 40 and w >= 20 and h >= 10 and 1.2 <= w / max(h, 1) <= 8.0:
            ax1 = xmin + x0 + 2
            ay1 = ymin + y0 + 2
            ax2 = xmax + x0 - 2
            ay2 = ymax + y0 - 2
            comps.append({"n": n, "x1": ax1, "y1": ay1, "x2": ax2, "y2": ay2,
                          "cx": (ax1 + ax2) / 2.0, "cy": (ay1 + ay2) / 2.0,
                          "w": w, "h": h})
    comps.sort(key=lambda c: c["w"], reverse=True)
    return comps


def track(frames, offset):
    n = len(frames)
    H = frames[0].shape[0]
    comps = [red_components(f) for f in frames]

    def ok(c):
        return TEXT_H <= c["cy"] <= 0.62 * H

    seed_i = seed = None
    for i in sorted(range(n), key=lambda i: abs(i - offset)):
        cands = [c for c in comps[i] if ok(c)]
        if cands:
            cands.sort(key=lambda c: c["w"], reverse=True)
            seed_i, seed = i, cands[0]
            break
    if seed is None:
        return [None] * n

    chosen = [None] * n
    chosen[seed_i] = seed

    def step(i, prev, wide):
        best = None
        bestd = 1e9
        gate_x = (1.6 if wide else 1.0) * max(prev["w"], 1.0)
        gate_y = (2.0 if wide else 1.4) * max(prev["h"], 1.0)
        for c in comps[i]:
            if not ok(c):
                continue
            dx = abs(c["cx"] - prev["cx"])
            dy = abs(c["cy"] - prev["cy"])
            if dx > gate_x or dy > gate_y:
                continue
            r = c["w"] / max(prev["w"], 1e-6)
            if not (0.55 <= r <= 1.80):
                continue
            cost = dx / gate_x + 0.6 * dy / max(prev["h"], 1.0) + 1.5 * abs(np.log(r))
            if cost < bestd:
                bestd, best = cost, c
        return best

    prev = seed
    for i in range(seed_i + 1, n):
        c = step(i, prev, False)
        if c is not None:
            chosen[i] = c
            prev = c
    prev = seed
    for i in range(seed_i - 1, -1, -1):
        c = step(i, prev, False)
        if c is not None:
            chosen[i] = c
            prev = c
    return chosen


def lane_runs(frame, y2, cx, bw):
    """Bright lane-marking column positions at the subject's contact row."""
    H = frame.shape[0]
    ylo = max(0, int(y2) - 8)
    yhi = min(H - 1, int(y2) + 2)
    if yhi - ylo < 3:
        return []
    band = frame[ylo:yhi + 1]
    chan_min = band.min(axis=2).astype(np.int16)
    bright = chan_min > 105
    cols = bright.sum(axis=0)
    thresh = max(2, int(0.35 * bright.shape[0]))
    xs = np.nonzero(cols >= thresh)[0]
    if xs.size == 0:
        return []
    runs = []
    start = prev = int(xs[0])
    for x in xs[1:]:
        x = int(x)
        if x - prev <= 3:
            prev = x
            continue
        runs.append((start, prev))
        start = prev = x
    runs.append((start, prev))
    out = []
    for a, b in runs:
        w = b - a + 1
        if 3 <= w <= 90:
            center = (a + b) / 2.0
            if abs(center - cx) <= 6.0 * bw:
                out.append(center)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="clips per class")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    samples = json.loads((S2 / "samples.json").read_text(encoding="utf-8"))
    pos = [s for s in samples if s["label"] == 1]
    neg = [s for s in samples if s["label"] == 0]
    picked = pos[:args.n] + neg[:args.n]

    for s in picked:
        cid = s["clip_id"]
        frames = [np.asarray(f) for f in iio.imiter(S2 / "clips" / f"{cid}.mp4", plugin="FFMPEG")]
        n = len(frames)
        H, W = frames[0].shape[:2]
        offset = int(s["f1"]) - int(s["clip_start"])
        boxes = track(frames, offset)
        valid = [b for b in boxes if b is not None]
        if not valid:
            print(f"{cid}: NO BOX", flush=True)
            continue
        bw = float(np.median([b["w"] for b in valid]))

        rows = []
        for i in range(n):
            b = boxes[i]
            if b is None:
                rows.append(None)
                continue
            centers = lane_runs(frames[i], b["y2"], b["cx"], bw)
            left = [c for c in centers if c < b["cx"] - 0.20 * bw]
            right = [c for c in centers if c > b["cx"] + 0.20 * bw]
            lc = (left[-1] + right[0]) / 2.0 if (left and right) else None
            rows.append({"cx": b["cx"], "y2": b["y2"], "w": b["w"],
                         "centers": centers, "lane_center": lc,
                         "y_cw": (b["cx"] - lc) / bw if lc is not None else None,
                         "on_mark": min([abs(c - b["cx"]) for c in centers], default=1e9) < 0.30 * bw})

        both = [r for r in rows if r and r["y_cw"] is not None]
        frac = len(both) / max(n, 1)

        # crop window, per clip fixed width driven by the median box width
        Wc = int(round(CROP_K * bw))
        Wc = int(np.clip(Wc, 160, W))
        Hc = int(round(Wc * PANEL[1] / PANEL[0]))
        picks = sorted({round(i * (n - 1) / (COUNT - 1)) for i in range(COUNT)})
        sheet = Image.new("RGB", (PANEL[0] * 4, PANEL[1] * 2), (0, 0, 0))
        draw = ImageDraw.Draw(sheet)
        for slot, idx in enumerate(picks):
            f = frames[idx].copy()
            f[:TEXT_H, :] = 0
            b = boxes[idx]
            if b is None:
                cx = int(np.median([v["cx"] for v in valid]))
                X1 = int(np.clip(cx - Wc // 2, 0, W - Wc))
                Y1 = 0
            else:
                X1 = int(np.clip(b["cx"] - Wc // 2, 0, W - Wc))
                Y1 = int(np.clip(b["y2"] - BOX_BOTTOM_FRAC * Hc, 0, H - Hc))
            crop = f[Y1:Y1 + Hc, X1:X1 + Wc]
            im = Image.fromarray(crop).resize(PANEL, Image.LANCZOS)
            px = (slot % 4) * PANEL[0]
            py = (slot // 4) * PANEL[1]
            sheet.paste(im, (px, py))
            sx = PANEL[0] / Wc
            sy = PANEL[1] / Hc
            r = rows[idx]
            if r is not None:
                for c in r["centers"]:
                    x = px + (c - X1) * sx
                    if px <= x <= px + PANEL[0]:
                        draw.line([(x, py), (x, py + PANEL[1])], fill=(0, 255, 0), width=1)
                if r["lane_center"] is not None:
                    x = px + (r["lane_center"] - X1) * sx
                    draw.line([(x, py), (x, py + PANEL[1])], fill=(0, 160, 255), width=2)
                if b is not None:
                    draw.line([(px, py + (b["y2"] - Y1) * sy),
                               (px + PANEL[0], py + (b["y2"] - Y1) * sy)], fill=(255, 0, 255), width=1)
            if b is not None:
                bx1 = px + (b["x1"] - X1) * sx
                bx2 = px + (b["x2"] - X1) * sx
                by1 = py + (b["y1"] - Y1) * sy
                by2 = py + (b["y2"] - Y1) * sy
                draw.rectangle([bx1, by1, bx2, by2], outline=(255, 0, 0), width=2)
            draw.rectangle([px, py, px + 250, py + 20], fill=(0, 0, 0))
            tag = f"{idx} y={'-' if r is None or r['y_cw'] is None else round(r['y_cw'], 2)}"
            draw.text((px + 4, py + 4), tag, fill=(255, 255, 0))
        sheet.save(OUT / f"{cid}.png")

        seq = [(i, None if r is None else r["y_cw"]) for i, r in enumerate(rows)]
        pts = [(i, v) for i, v in seq if v is not None]
        step = max(1, len(pts) // 8)
        brief = " ".join(f"{i}:{v:+.2f}" for i, v in pts[::step])
        print(f"{cid} label={s['label']} stratum={s.get('stratum')} bw={bw:.0f} "
              f"ruler={frac:.2f} y_cw=[{brief}]", flush=True)


if __name__ == "__main__":
    main()

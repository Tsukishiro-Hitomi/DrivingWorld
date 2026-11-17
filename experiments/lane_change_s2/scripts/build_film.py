"""Build subject-crop filmstrips for the S2 lane-change set."""
import json
import numpy as np
from pathlib import Path
import imageio.v3 as iio
from PIL import Image, ImageDraw

from _paths import REPO, S2  # noqa: E402
OUT = S2 / "sheets_film"
TEXT_H = 60
PANEL = (480, 240)
COUNT = 8


def red_components(frame):
    arr = frame.astype(np.int16)
    m = (arr[..., 0] > 150) & (arr[..., 1] < 90) & (arr[..., 2] < 90)
    m[:TEXT_H, :] = False
    ys, xs = np.nonzero(m)
    coords = set(zip(ys.tolist(), xs.tolist()))
    comps = []
    while coords:
        st = coords.pop(); stack = [st]; n = 1
        ymin = ymax = st[0]; xmin = xmax = st[1]
        while stack:
            y, x = stack.pop()
            ymin = min(ymin, y); ymax = max(ymax, y); xmin = min(xmin, x); xmax = max(xmax, x)
            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if (ny, nx) in coords:
                    coords.remove((ny, nx)); stack.append((ny, nx)); n += 1
        w = xmax - xmin + 1; h = ymax - ymin + 1
        if n >= 25 and h >= 6 and 1.2 <= w / h <= 8:
            comps.append({"n": n, "x1": xmin, "y1": ymin, "x2": xmax, "y2": ymax,
                          "cx": (xmin + xmax) / 2.0, "cy": (ymin + ymax) / 2.0, "w": w, "h": h})
    return comps


def track(frames, offset):
    n = len(frames); H = frames[0].shape[0]
    comps = [red_components(f) for f in frames]

    def ok(c):
        return TEXT_H <= c["cy"] <= 0.62 * H

    seed_i = None; seed = None
    for i in sorted(range(n), key=lambda i: abs(i - offset)):
        cands = [c for c in comps[i] if ok(c)]
        if cands:
            cands.sort(key=lambda c: c["n"], reverse=True)
            seed_i, seed = i, cands[0]; break
    if seed is None:
        return [None] * n
    chosen = [None] * n; chosen[seed_i] = seed

    def pick(i, prev):
        cands = [c for c in comps[i] if ok(c)]
        if not cands:
            return None
        c = min(cands, key=lambda c: abs(c["cx"] - prev["cx"]) + abs(c["cy"] - prev["cy"]))
        return c if abs(c["cx"] - prev["cx"]) + abs(c["cy"] - prev["cy"]) <= 250 else None

    prev = seed
    for i in range(seed_i + 1, n):
        c = pick(i, prev)
        if c:
            chosen[i] = c; prev = c
    prev = seed
    for i in range(seed_i - 1, -1, -1):
        c = pick(i, prev)
        if c:
            chosen[i] = c; prev = c
    valid = [i for i, c in enumerate(chosen) if c is not None]
    for i in range(n):
        if chosen[i] is None:
            j = min(valid, key=lambda j: abs(j - i))
            chosen[i] = chosen[j]
    return chosen


def build(clip_id, offset):
    frames = [np.asarray(f) for f in iio.imiter(S2 / "clips" / f"{clip_id}.mp4", plugin="FFMPEG")]
    n = len(frames); H, W = frames[0].shape[:2]
    boxes = track(frames, offset)
    valid = [b for b in boxes if b is not None]
    if valid:
        cxs = [b["cx"] for b in valid]; cys = [b["cy"] for b in valid]
        bws = [b["w"] for b in valid]; bhs = [b["h"] for b in valid]
        bw = float(np.median(bws)); bh = float(np.median(bhs))
        cx = (min(cxs) + max(cxs)) / 2.0; cy = float(np.median(cys))
        span_x = max(cxs) - min(cxs)
        cw = int(np.clip(max(14 * bw, span_x + 6 * bw), 720, 1800))
        ch = int(np.clip(8 * bh, 320, 600))
    else:
        cw, ch = int(W * 0.6), int(H * 0.55); cx, cy = W / 2, H * 0.30
    X1 = int(np.clip(cx - cw / 2, 0, W - cw)); Y1 = int(np.clip(cy - ch / 2, 0, H - ch))
    picks = sorted({round(i * (n - 1) / (COUNT - 1)) for i in range(COUNT)})
    sheet = Image.new("RGB", (PANEL[0] * 4, PANEL[1] * 2), (0, 0, 0))
    draw = ImageDraw.Draw(sheet)
    for slot, idx in enumerate(picks):
        f = frames[idx].copy(); f[:TEXT_H, :] = 0
        crop = f[Y1:Y1 + ch, X1:X1 + cw]
        im = Image.fromarray(crop).resize(PANEL, Image.LANCZOS)
        x = (slot % 4) * PANEL[0]; y = (slot // 4) * PANEL[1]
        sheet.paste(im, (x, y))
        draw.rectangle([x, y, x + 190, y + 22], fill=(0, 0, 0))
        draw.text((x + 4, y + 4), f"{slot + 1}/{COUNT} t={idx / 10:.1f}s", fill=(255, 255, 0))
    sheet.save(OUT / f"{clip_id}.png")
    return len(valid)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    samples = json.loads((S2 / "samples.json").read_text(encoding="utf-8"))
    done = {p.stem for p in OUT.glob("*.png")}
    todo = [s for s in samples if s["clip_id"] not in done]
    print(f"todo {len(todo)}/{len(samples)}", flush=True)
    for i, s in enumerate(todo):
        offset = int(s["f1"]) - int(s["clip_start"])
        try:
            nv = build(s["clip_id"], offset)
            print(f"[{i+1}/{len(todo)}] {s['clip_id']} boxes={nv}", flush=True)
        except Exception as e:
            print(f"[{i+1}/{len(todo)}] FAIL {s['clip_id']}: {type(e).__name__}: {e}", flush=True)
    print("DONE", len(list(OUT.glob('*.png'))), flush=True)


if __name__ == "__main__":
    main()

"""Load the real checkpoint and check that the depth is metric."""
from pathlib import Path

import numpy as np
import imageio.v3 as iio
import torch
from depth_anything_3.api import DepthAnything3

from _paths import REPO, S2  # noqa: E402
MODEL_DIR = REPO / "models/da3metric-large"

device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda")
model = DepthAnything3.from_pretrained(str(MODEL_DIR)).to(device=device)
model.eval()
print("loaded", model.__class__.__name__, "model_name", getattr(model, "model_name", "?"))

frames = [np.asarray(f) for f in
          iio.imiter(S2 / "clips" / "pos_record1drive1_03270.mp4", plugin="FFMPEG")]
pred = model.inference([frames[10]], process_res=504)
z = pred.depth[0]
print("depth", z.shape, "range", round(float(np.nanmin(z)), 2),
      round(float(np.nanmedian(z)), 2), round(float(np.nanmax(z)), 2))
print("is_metric", pred.is_metric, "scale_factor", pred.scale_factor)
print("conf", None if pred.conf is None else (pred.conf.shape, float(pred.conf.mean())))

# the subject at this frame, per our detector, and its GT depth
import json
det_line = None
for line in (S2 / "results/s2_dets.jsonl").open():
    if not line.strip():
        continue
    r = json.loads(line)
    if r["clip_id"] == "pos_record1drive1_03270":
        det_line = r["frames"]
        break
rec = det_line[10]
red = rec["red"]
cand = [d for d in rec["dets"]
        if d[2] <= red["cx"] <= d[4] and d[3] - 0.35 * (d[5] - d[3]) <= red["cy"] <= d[5]]
print("candidate dets:", len(cand))
if cand:
    x1, y1, x2, y2 = cand[0][2], cand[0][3], cand[0][4], cand[0][5]
    h, w = z.shape
    sx, sy = w / 1920, h / 600
    patch = z[int(y1 * sy):int(y2 * sy), int(x1 * sx):int(x2 * sx)]
    print("subject box depth (median):", round(float(np.median(patch)), 2), "m")
    print("box", x1, y1, x2, y2)

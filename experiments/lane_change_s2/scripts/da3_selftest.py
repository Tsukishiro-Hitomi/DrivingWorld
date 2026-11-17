"""De-risk the DA3 call path with random weights (no checkpoint needed).

Verifies: registry preset name, model construction, the inference signature,
the output shapes, and how the depth map's resolution maps back to the original
1920x600 frame — everything except loading the real checkpoint.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/root/miniconda3/envs/h3-ref2va/lib/python3.11/site-packages")
import torch
from depth_anything_3.api import DepthAnything3
from depth_anything_3.registry import MODEL_REGISTRY

print("registry:", sorted(MODEL_REGISTRY.keys()))

device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda")
model = DepthAnything3(model_name="da3metric-large").to(device=device)
model.eval()
n = sum(p.numel() for p in model.parameters())
print(f"built da3metric-large: {n/1e6:.1f}M params")

from _paths import S2  # noqa: E402
import imageio.v3 as iio
frames = [np.asarray(f) for f in
          iio.imiter(S2 / "clips" / "pos_record1drive1_03270.mp4", plugin="FFMPEG")]
print("frame", frames[10].shape, frames[10].dtype)

pred = model.inference([frames[10]], process_res=504)
print("depth", pred.depth.shape, pred.depth.dtype)
print("is_metric", pred.is_metric, "scale_factor", pred.scale_factor)
print("conf", None if pred.conf is None else pred.conf.shape)
print("intrinsics", None if pred.intrinsics is None else pred.intrinsics[0])
print("processed_images", None if pred.processed_images is None else pred.processed_images.shape)
z = pred.depth[0]
print("depth range", float(np.nanmin(z)), float(np.nanmedian(z)), float(np.nanmax(z)))
h, w = z.shape
print("depth h,w", h, w, "-> orig 600x1920, sx", w / 1920, "sy", h / 600)
print("OK")

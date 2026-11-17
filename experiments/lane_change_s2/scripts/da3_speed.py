"""Measure steady-state DA3 throughput on this GPU."""
import time
from pathlib import Path

import numpy as np
import imageio.v3 as iio
import torch
from depth_anything_3.api import DepthAnything3

from _paths import S2  # noqa: E402
frames = [np.asarray(f) for f in
          iio.imiter(S2 / "clips" / "pos_record1drive1_03270.mp4", plugin="FFMPEG")]
device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda")
model = DepthAnything3(model_name="da3metric-large").to(device=device)
model.eval()

for tag, ctx in (("fp32", None), ("autocast-fp16", torch.autocast("cuda", dtype=torch.float16))):
    t0 = time.time()
    n = 6
    for i in range(n):
        if ctx is None:
            pred = model.inference([frames[i * 5]], process_res=504)
        else:
            with ctx:
                pred = model.inference([frames[i * 5]], process_res=504)
    dt = (time.time() - t0) / n
    print(f"{tag}: {dt*1000:.0f} ms/frame -> {221*40*dt/60:.1f} min for 221x40 on one GPU")

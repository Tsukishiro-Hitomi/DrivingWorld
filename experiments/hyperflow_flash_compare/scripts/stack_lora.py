from pathlib import Path

import imageio.v3 as iio
from PIL import Image, ImageDraw

EXP = Path("/xs-fd-nas/workspace_zhaoxinrui/driving-world/experiments/hyperflow_flash_compare")
rows = [
    ("h3_8step", "H3 base, 8 steps, no LoRA (696 s)"),
    ("h3_30step", "H3 base, 30 steps, no LoRA (1752 s)"),
    ("turbo_8step", "H3 + 8-step LoRA, alpha=1.0, shift=6 (634 s)"),
    ("lora_alpha00625", "H3 + 8-step LoRA, alpha=6.25e-2, shift=6 (838 s)"),
    ("lora_alpha00625_defaultshift", "H3 + 8-step LoRA, alpha=6.25e-2, default shift (711 s)"),
]
T = 3.5
target = round(T * 24)


def frame_at(video: Path, index: int) -> Image.Image:
    frame = None
    for i, frame in enumerate(iio.imiter(video, plugin="FFMPEG")):
        if i >= index:
            break
    return Image.fromarray(frame).convert("RGB")


for tag, _ in rows:
    video = EXP / "videos" / f"{tag}.mp4"
    out = EXP / "frames" / f"{tag}_t{T}.png"
    if video.is_file() and not out.is_file():
        frame_at(video, target).save(out)
        print("wrote", out.name)

ims = [Image.open(EXP / "frames" / f"{tag}_t{T}.png").convert("RGB") for tag, _ in rows]
w, h = ims[0].size
bar = 30
sheet = Image.new("RGB", (w, (h + bar) * len(ims)), "white")
draw = ImageDraw.Draw(sheet)
for i, (im, (_, label)) in enumerate(zip(ims, rows)):
    top = i * (h + bar)
    draw.text((10, top + 9), label, fill="black")
    sheet.paste(im, (0, top + bar))
sheet.save(EXP / "frames" / "compare_lora_t3.5.png")
print("ok", sheet.size)

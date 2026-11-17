from PIL import Image, ImageDraw

EXP = "/xs-fd-nas/workspace_zhaoxinrui/driving-world/experiments/hyperflow_flash_compare"
rows = [
    ("h3_8step", "H3 base, 8 steps"),
    ("flash_6step", "Flash 6 steps, prompt with <Picture 1>/<Video 1>"),
    ("flash_6step_h3prompt", "Flash 6 steps, pipeline prompt"),
    ("flash_8step", "Flash 8 steps, prompt with <Picture 1>/<Video 1>"),
]
ims = [Image.open(f"{EXP}/frames/{n}_t3.5.png").convert("RGB") for n, _ in rows]
w, h = ims[0].size
bar = 34
sheet = Image.new("RGB", (w, (h + bar) * len(ims)), "white")
draw = ImageDraw.Draw(sheet)
for i, (im, (_, label)) in enumerate(zip(ims, rows)):
    top = i * (h + bar)
    draw.text((10, top + 10), label, fill="black")
    sheet.paste(im, (0, top + bar))
sheet.save(f"{EXP}/frames/compare_all_t3.5.png")
print("ok", sheet.size)

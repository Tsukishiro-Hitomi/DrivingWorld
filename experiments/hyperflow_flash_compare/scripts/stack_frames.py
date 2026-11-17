from PIL import Image

EXP = "/xs-fd-nas/workspace_zhaoxinrui/driving-world/experiments/hyperflow_flash_compare"
names = [("h3_8step", "H3 8-step"), ("flash_6step", "Flash 6-step"), ("flash_4step", "Flash 4-step")]
for t in ("1.5", "3.5"):
    ims = [Image.open(f"{EXP}/frames/{n}_t{t}.png").convert("RGB") for n, _ in names]
    w, h = ims[0].size
    sheet = Image.new("RGB", (w, h * 3), "white")
    for i, im in enumerate(ims):
        sheet.paste(im, (0, i * h))
    sheet.save(f"{EXP}/frames/compare_t{t}.png")
    print(f"compare_t{t}.png", sheet.size)

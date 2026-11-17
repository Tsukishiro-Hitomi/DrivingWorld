from PIL import Image

src = "/xs-fd-nas/workspace_zhaoxinrui/driving-world/experiments/reference_h3_only_longtail_v1.png"
dst = "/xs-fd-nas/workspace_zhaoxinrui/driving-world/experiments/hyperflow_flash_compare/frames/reference_832x480.png"
Image.open(src).convert("RGB").resize((832, 480), Image.Resampling.LANCZOS).save(dst)
print(dst)

"""Decode and compare videos without changing their time base."""

from pathlib import Path

def read_frames(path, frames=124, fps=24, width=832, height=480):
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(path))
    result = []
    try:
        rate = cap.get(cv2.CAP_PROP_FPS)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape != (height, width, 3):
                raise ValueError(f"Expected {width}x{height}, received {frame.shape}: {path}")
            result.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if len(result) != frames or abs(rate - fps) > .001:
        raise ValueError(f"Expected {frames} frames at {fps} FPS, received {len(result)} at {rate}: {path}")
    return np.stack(result)


def sample_sheet(path, source, output=None, fps=24, source_label="SOURCE"):
    import numpy as np
    from PIL import Image, ImageDraw

    indices = np.linspace(0, len(source) - 1, 6).round().astype(int)
    sheet = Image.new("RGB", (320 * len(indices), 216 * (1 if output is None else 2)), "#141923")
    draw = ImageDraw.Draw(sheet)
    for row, frames in enumerate([source] if output is None else [source, output]):
        for col, i in enumerate(indices):
            picture = Image.fromarray(frames[i])
            picture.thumbnail((320, 184))
            x, y = col * 320, row * 216
            sheet.paste(picture, (x, y + 28))
            draw.text((x + 4, y + 7), f"{source_label if row == 0 else 'OUTPUT'} {i} / {i/fps:.2f}s", fill="white")
    sheet.save(path, quality=92)


def comparison(path, source, output, label, fps=24):
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    w = source.shape[2]
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), 24) if font_path.exists() else ImageFont.load_default()
    banner = Image.new("RGB", (w * 2, 48), "#141923")
    draw = ImageDraw.Draw(banner)
    draw.text((16, 10), "Original", font=font, fill="white")
    draw.text((w + 16, 10), label, font=font, fill="white")
    with imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8,
                            macro_block_size=1, pixelformat="yuv420p") as writer:
        for a, b in zip(source, output):
            writer.append_data(np.concatenate([np.asarray(banner), np.concatenate([a, b], axis=1)]))

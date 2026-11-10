"""Decode a new scene and provide sampled visual evidence without fake source references."""

from PIL import Image, ImageDraw
import numpy as np

from ..qwen import run_worker
from ..runtime import dump, load, sha
from ..video import read_frames, sample_sheet
from .contracts import evaluate_review


def verify(out):
    config, plan = load(out / "config.json"), load(out / "plan.json")
    generation = load(out / "generation.json")
    if (generation["generated_sha256"] != sha(out / "generated.mp4")
            or any(generation.get(k) != v for k, v in config.items())
            or config["plan_sha256"] != sha(out / "plan.json")):
        raise ValueError("Full-text generation/plan binding changed")
    frames = read_frames(out / "generated.mp4")
    sample_sheet(out / "review_frames.jpg", frames, source_label="OUTPUT frame")
    # Adjacent triplets reveal local changes; gaps remain explicitly unverified.
    indices = sorted({i for c in np.linspace(0, len(frames) - 1, 10).round().astype(int)
                      for i in range(max(0, int(c) - 1), min(len(frames), int(c) + 2))})
    picture = Image.new("RGB", (416 * 4, 268 * ((len(indices) + 3) // 4)), "#141923")
    draw = ImageDraw.Draw(picture)
    for n, index in enumerate(indices):
        x, y = n % 4 * 416, n // 4 * 268
        image = Image.fromarray(frames[index])
        image.thumbnail((416, 240))
        picture.paste(image, (x, y + 28))
        draw.text((x + 4, y + 7), f"OUTPUT frame {index} / {index / config['fps']:.3f}s", fill="white")
    picture.save(out / "review_sequence.jpg", quality=94)
    observed = sorted(set(indices) | set(np.linspace(0, len(frames) - 1, 6).round().astype(int).tolist()))
    manifest = dict(observed_frames=observed, frame_count=len(frames), fps=config["fps"], source_video=None,
        reference_image_count=0, reference_video_count=0, keyframe_count=0, all_frame_visual_verification=False,
        generated_sha256=sha(out / "generated.mp4"), sheet_sha256=sha(out / "review_frames.jpg"),
        sequence_sha256=sha(out / "review_sequence.jpg"))
    dump(out / "review_manifest.json", manifest)
    dump(out / "technical_checks.json", dict(frames=len(frames), fps=config["fps"], width=config["width"],
        height=config["height"], duration_s=len(frames) / config["fps"], generated_sha256=sha(out / "generated.mp4"),
        input_mode="T2VA", checkpoint_variant="FL2VA", source_video=None,
        note="Media integrity does not prove scene quality, physics or full-sequence temporal consistency."))
    if not (out / "visual_review.json").exists():
        if list(out.glob("full_text_review_response_*.json")):
            raise ValueError("A previous review response is already recorded; no hidden review retry")
        run_worker("agent_loop.full_text", "review", out, "review.log")
    result = load(out / "visual_review.json")
    if (result["generated_sha256"] != sha(out / "generated.mp4") or result["plan_sha256"] != sha(out / "plan.json")
            or result["manifest_sha256"] != sha(out / "review_manifest.json")):
        raise ValueError("Full-text review belongs to different artifacts")
    computed = evaluate_review(result.get("raw_result", result), plan, manifest)
    fields = ("review_valid", "sampled_verdict", "acceptance_verdict")
    fields += ("effective_checks",) if computed["review_valid"] else ("review_error", "raw_result")
    if any(result[key] != computed[key] for key in fields):
        raise ValueError("Full-text acceptance does not match its evidence gates")
    return result

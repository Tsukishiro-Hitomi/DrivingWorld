"""Technical decoding and event-aware visual evidence, with explicit acceptance limits."""

from pathlib import Path
import json

from ..acceptance import assess
from ..qwen import run_worker
from ..runtime import dump, load, sha
from ..video import comparison, read_frames, sample_sheet
from .contracts import CHECKS


def assess_insertion(result, manifest):
    return assess(result, CHECKS, observed_frames=manifest["observed_frames"], frame_count=manifest["frame_count"],
                  full_sequence_ids=("temporal_continuity",))


def assess_recorded(out):
    """Recover a recorded review after a format adapter fix, with no model/generation call."""
    manifest = load(out / "review_manifest.json")
    original = load(out / "review_input.json")
    payload = json.loads(original[1]["content"][-1]["text"])
    if (payload["evidence"] != manifest or payload["plan"] != load(out / "plan.json")
            or payload["request"] != load(out / "request.json")):
        raise ValueError("Recorded review inputs differ from current artifacts")
    for key, file in [("generated_sha256", "generated.mp4"), ("sheet_sha256", "review_frames.jpg"),
                      ("event_sheet_sha256", "review_event_frames.jpg")]:
        if manifest[key] != sha(out / file):
            raise ValueError("Recorded review evidence changed: " + file)
    raw_file = sorted(out.glob("insert_review_response_*.json"))[-1]
    raw = load(raw_file)["text"]
    parsed = json.loads(raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip())
    assessed = assess_insertion(parsed, manifest)
    assessed.update(generated_sha256=sha(out / "generated.mp4"), plan_sha256=sha(out / "plan.json"),
                    manifest_sha256=sha(out / "review_manifest.json"),
                    raw_response_file=raw_file.name, raw_response_sha256=sha(raw_file),
                    assessment_source="recorded model response; no generation or model retry",
                    review_type="Qwen qualitative sampled-frame review, with limitation and coverage gates")
    dump(out / "visual_review.json", assessed)
    return assessed


def event_sheet(out, frames, plan, fps):
    import numpy as np
    from PIL import Image, ImageDraw

    event = plan["event"]
    start, end = event["start_frame"], event["end_frame"]
    indices = set(np.linspace(max(0, start - 3), min(len(frames) - 1, end + 3), 16).round().astype(int).tolist())
    for center in (start, (start + end) // 2, end):
        indices.update(i for i in range(center - 2, center + 3) if 0 <= i < len(frames))
    indices.update((0, len(frames) - 1))
    indices = sorted(indices)
    cols, tile_w, tile_h = 4, 416, 268
    sheet = Image.new("RGB", (cols * tile_w, ((len(indices) + cols - 1) // cols) * tile_h), "#141923")
    draw = ImageDraw.Draw(sheet)
    for number, i in enumerate(indices):
        x, y = number % cols * tile_w, number // cols * tile_h
        image = Image.fromarray(frames[i])
        image.thumbnail((416, 240))
        sheet.paste(image, (x, y + 28))
        draw.text((x + 4, y + 7), f"OUTPUT frame {i} / {i / fps:.3f}s", fill="white")
    sheet.save(out / "review_event_frames.jpg", quality=94)
    return indices


def verify(out):
    config, plan = load(out / "config.json"), load(out / "plan.json")
    if sha(Path(config["source"])) != config["source_sha256"] or sha(out / "plan.json") != config["plan_sha256"]:
        raise ValueError("Insertion input or plan changed before verification")
    generation = load(out / "generation.json")
    if (generation["generated_sha256"] != sha(out / "generated.mp4")
            or any(generation.get(k) != v for k, v in config.items())):
        raise ValueError("Insertion generation differs from its input manifest")
    source, frames = read_frames(config["source"]), read_frames(out / "generated.mp4")
    comparison(out / "comparison.mp4", source, frames, "Object Insertion")
    sample_sheet(out / "review_frames.jpg", source, frames)
    indices = event_sheet(out, frames, plan, config["fps"])
    import numpy as np

    observed = sorted(set(indices) | set(np.linspace(0, len(frames) - 1, 6).round().astype(int).tolist()))
    manifest = dict(observed_frames=observed, frame_count=len(frames), fps=config["fps"],
        event_sheet_frames=indices, comparison_sheet_frames=np.linspace(0, len(frames) - 1, 6).round().astype(int).tolist(),
        generated_sha256=sha(out / "generated.mp4"), source_sha256=config["source_sha256"],
        sheet_sha256=sha(out / "review_frames.jpg"), event_sheet_sha256=sha(out / "review_event_frames.jpg"),
        coverage="sampled_frames", all_frame_visual_verification=False)
    dump(out / "review_manifest.json", manifest)
    dump(out / "technical_checks.json", dict(frames=len(frames), fps=config["fps"], width=config["width"],
        height=config["height"], duration_s=len(frames) / config["fps"], generated_sha256=sha(out / "generated.mp4"),
        note="Full decoding validates media integrity, not all-frame visual continuity or metric geometry."))
    if not (out / "visual_review.json").exists():
        if list(out.glob("insert_review_response_*.json")):
            raise ValueError("An invalid visual review is already recorded; automatic re-review is disabled")
        run_worker("agent_loop.insert", "review", out, "review.log")
    result = load(out / "visual_review.json")
    if (result["manifest_sha256"] != sha(out / "review_manifest.json")
            or result["plan_sha256"] != sha(out / "plan.json")
            or result["generated_sha256"] != sha(out / "generated.mp4")):
        raise ValueError("Insertion review belongs to different evidence")
    recomputed = assess_insertion(result, manifest)
    if any(result[key] != recomputed[key] for key in ("effective_checks", "acceptance_verdict", "sampled_verdict")):
        raise ValueError("Insertion acceptance gates differ from their recorded result")
    return result

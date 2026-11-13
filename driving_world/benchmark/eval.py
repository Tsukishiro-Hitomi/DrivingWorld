"""Evaluate a VLM's event recognition on the real-video benchmark.

Reads ``manifest.jsonl`` (built by :mod:`driving_world.benchmark.build`), asks
the VLM once per clip which of the three event kinds occur, and compares the
answer against the trajectory-derived ground-truth labels.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import imageio.v3 as iio
from PIL import Image

KINDS = [
    "vehicle_lane_change",
    "pedestrian_crossing",
    "vehicle_hard_brake",
]

KIND_ITEMS = {
    "vehicle_lane_change": "A vehicle changes lanes (moves from one lane to an adjacent lane).",
    "pedestrian_crossing": "A pedestrian walks into or crosses the road ahead of the camera.",
    "vehicle_hard_brake": "A vehicle ahead brakes hard (decelerates noticeably).",
}

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _load_model(model_path: str):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda:0", local_files_only=True
    ).eval()
    return processor, model


def sample_frames(video: Path, n: int = 12) -> list[Image.Image]:
    frames = list(iio.imiter(video, plugin="FFMPEG"))
    indices = sorted({round(i * (len(frames) - 1) / (n - 1)) for i in range(n)})
    return [Image.fromarray(frames[i]) for i in indices]


def _prompt() -> str:
    items = "\n".join(f"{i + 1}. {KIND_ITEMS[kind]}" for i, kind in enumerate(KINDS))
    fields = ", ".join(f'"{kind}": <true or false>' for kind in KINDS)
    return (
        "Watch this driving video. For each item, answer true if it occurs in the "
        f"video and false otherwise.\n{items}\nOutput only a JSON object: {{{fields}}}"
    )


def _parse(raw: str) -> dict[str, bool]:
    match = _JSON_RE.search(raw)
    if not match:
        raise ValueError(f"no JSON object in response: {raw[:120]!r}")
    data = json.loads(match.group(0))
    return {kind: bool(data.get(kind, False)) for kind in KINDS}


def evaluate(model_path: str, manifest_path: Path) -> dict:
    processor, model = _load_model(model_path)
    manifest = [
        json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]

    per_kind = {kind: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for kind in KINDS}
    rows = []
    for entry in manifest:
        video = manifest_path.parent / entry["video"]
        images = sample_frames(video)
        raw = _ask(processor, model, _prompt(), images)
        predicted = _parse(raw)
        labels = set(entry["labels"])
        row = {"clip_id": entry["clip_id"], "labels": sorted(labels), "predicted": predicted}
        rows.append(row)
        for kind in KINDS:
            actual = kind in labels
            pred = predicted[kind]
            if actual and pred:
                per_kind[kind]["tp"] += 1
            elif actual and not pred:
                per_kind[kind]["fn"] += 1
            elif not actual and pred:
                per_kind[kind]["fp"] += 1
            else:
                per_kind[kind]["tn"] += 1

    summary = {}
    for kind in KINDS:
        stats = per_kind[kind]
        precision = stats["tp"] / (stats["tp"] + stats["fp"]) if stats["tp"] + stats["fp"] else 0.0
        recall = stats["tp"] / (stats["tp"] + stats["fn"]) if stats["tp"] + stats["fn"] else 0.0
        accuracy = (stats["tp"] + stats["tn"]) / sum(stats.values())
        summary[kind] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "accuracy": round(accuracy, 3),
            **stats,
        }
    return {"summary": summary, "rows": rows}


def _ask(processor, model, prompt: str, images: list[Image.Image]) -> str:
    import torch

    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": prompt})
    text = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=images, padding=True, return_tensors="pt").to(
        "cuda:0"
    )
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    return processor.batch_decode(
        output[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
    )[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.model_path, args.manifest)
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

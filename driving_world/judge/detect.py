"""Open-vocabulary grounding detectors for photorealistic subject tracking."""

from __future__ import annotations

import re

from PIL import Image

# Qwen2.5-VL emits grounded boxes either as a JSON list of ``bbox_2d`` (absolute
# pixel coordinates) or as ``<|box_start|>(x1,y1),(x2,y2)<|box_end|>`` on a
# 1000x1000 grid. Both are parsed and returned as ``(bbox_xyxy, score)``.
_BBOX_2D_RE = re.compile(
    r'"bbox_2d"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]'
)
_BOX_START_RE = re.compile(
    r"\((\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\),(\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\)"
)
_GRID = 1000.0


class Qwen25VLDetector:
    """Ground a subject with Qwen2.5-VL and return its bounding box in pixels."""

    name = "qwen2.5-vl-grounding"

    def __init__(self, model_path: str, subject: str, device: str = "cuda:0") -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.subject = subject
        self.device = device
        self._torch = torch
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            local_files_only=True,
        ).eval()

    def _prompt(self) -> str:
        return f"Detect the {self.subject} in the image and output its bounding box."

    def detect(self, image: Image.Image) -> list[tuple[tuple[float, float, float, float], float]]:
        content = [
            {"type": "image", "image": image},
            {"type": "text", "text": self._prompt()},
        ]
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text], images=[image], padding=True, return_tensors="pt"
        ).to(self.device)
        with self._torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        raw = self.processor.batch_decode(
            output[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )[0]

        width, height = image.size
        boxes: list[tuple[tuple[float, float, float, float], float]] = []
        for match in _BBOX_2D_RE.finditer(raw):
            x1, y1, x2, y2 = (float(value) for value in match.groups())
            boxes.append(((x1, y1, x2, y2), 1.0))
        if boxes:
            return boxes
        for match in _BOX_START_RE.finditer(raw):
            x1, y1, x2, y2 = (float(value) for value in match.groups())
            boxes.append(
                (
                    (x1 / _GRID * width, y1 / _GRID * height, x2 / _GRID * width, y2 / _GRID * height),
                    1.0,
                )
            )
        return boxes

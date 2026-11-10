"""Qwen2.5-VL adapter used by the first Event Judge baseline."""

from __future__ import annotations

from typing import Protocol

from .sampling import VideoSample


class VisionLanguageModel(Protocol):
    """Minimal interface that makes alternative VLM backends interchangeable."""

    name: str

    def answer(self, sample: VideoSample, prompt: str) -> str:
        """Return the model's unprocessed text response."""


class Qwen25VL:
    """Local Qwen2.5-VL inference backend, loaded only when judging begins."""

    name = "qwen2.5-vl-7b-instruct"

    def __init__(self, model_path: str, device: str = "cuda:0") -> None:
        self.model_path = model_path
        self.device = device

    def answer(self, sample: VideoSample, prompt: str) -> str:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            local_files_only=True,
        ).eval()
        content = []
        for frame_order, frame in enumerate(sample.frames):
            content.extend(
                [
                    {"type": "text", "text": f"Frame {frame_order} at {frame.timestamp_seconds:.3f}s:"},
                    {"type": "image", "image": frame.image},
                ]
            )
        content.append({"type": "text", "text": prompt})
        text = processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        images = [frame.image for frame in sample.frames]
        inputs = processor(text=[text], images=images, padding=True, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=768, do_sample=False)
        return processor.batch_decode(output[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)[0]

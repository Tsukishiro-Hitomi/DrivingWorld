"""Run infer_h3.py with a LoRA hot-loaded onto the ref2va DiT.

Differences from ``waymo-world-model/outputs/infer_h3_turbo.py``: the LoRA alpha and the
flow shifts come from the environment, so the checkpoint's PEFT scaling (alpha / rank)
can be applied. The repo wrapper multiplied every LoRA up-projection by 1.0, which is 16x
the checkpoint's own 8/128 scaling.
"""

import os
import runpy

from h3_ref2va.core.vram.layers import LoRAHotLoadMixin
from h3_ref2va.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline

_original = MiniMaxH3Pipeline.from_pretrained


def _load_with_lora(cls, *args, **kwargs):
    pipe = _original.__func__(cls, *args, **kwargs) if hasattr(_original, "__func__") else _original(*args, **kwargs)
    path = os.environ["H3_LORA"]
    alpha = float(os.environ.get("H3_LORA_ALPHA", "1.0"))
    print("LOADING_LORA", path, "alpha", alpha, flush=True)
    pipe.load_lora(pipe.dit, path, alpha=alpha, hotload=True)
    count = sum(bool(m.lora_A_weights) for m in pipe.dit.modules() if isinstance(m, LoRAHotLoadMixin))
    print("LORA_PATCHED_MODULES", count, flush=True)
    if count < 100:
        raise RuntimeError(f"Insufficient LoRA modules patched: {count}")
    return pipe


MiniMaxH3Pipeline.from_pretrained = classmethod(_load_with_lora)

_shift = os.environ.get("H3_FLOW_SHIFT")
_audio_shift = os.environ.get("H3_AUDIO_FLOW_SHIFT")
_original_call = MiniMaxH3Pipeline.__call__


def _call_with_shift(self, *args, **kwargs):
    if _shift is not None:
        kwargs["flow_shift"] = float(_shift)
    if _audio_shift is not None:
        kwargs["audio_flow_shift"] = float(_audio_shift)
    return _original_call(self, *args, **kwargs)


MiniMaxH3Pipeline.__call__ = _call_with_shift

runpy.run_path(
    "/xs-fd-nas/workspace_zhaoxinrui/waymo-world-model/src/infer_h3.py",
    run_name="__main__",
)

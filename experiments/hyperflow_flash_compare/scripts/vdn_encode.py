"""Encode a ref2va-like prompt cache for VDN-H3 with the local MiniMax-H3 text encoder.

VDN resolves the text encoder through ``upstream_snapshot`` (Hugging Face), which this
host cannot reach. The same Qwen3-VL-32B weights already live in
``models/MiniMax-H3-base`` (text_encoder / tokenizer / processor).
"""

import sys

sys.path.insert(0, "/xs-fd-nas/workspace_zhaoxinrui/vdn-h3/vdn-minimax-h3")

import vdn_shim  # noqa: F401  (torch.accelerator shim for torch 2.5)

import src.paths as paths

LOCAL_BASE = "/xs-fd-nas/workspace_zhaoxinrui/models/MiniMax-H3-base"


def _local_snapshot(*subfolders):
    for subfolder in subfolders:
        assert __import__("os").path.isdir(f"{LOCAL_BASE}/{subfolder}"), subfolder
    return LOCAL_BASE


paths.upstream_snapshot = _local_snapshot

import src.inference.encode_keyframes as encode  # noqa: E402

encode.upstream_snapshot = _local_snapshot
encode.CANVAS_SHORT_EDGE = 480

if __name__ == "__main__":
    sys.exit(encode.main())

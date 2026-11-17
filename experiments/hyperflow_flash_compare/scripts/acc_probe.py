"""Check whether the diffsynth-style Acc-8Step LoRA binds with the repo's loader."""

import sys

import torch
from safetensors.torch import load_file

sys.path.insert(0, "/xs-fd-nas/workspace_zhaoxinrui/h3/src")
from h3_ref2va.utils.lora.minimax_h3 import MiniMaxH3LoRALoader

path = "/xs-fd-nas/workspace_zhaoxinrui/models/lora/MiniMax-H3-Ref2VA-Acc-8Step.safetensors"
sd = load_file(path)
loader = MiniMaxH3LoRALoader(device="cpu", torch_dtype=torch.bfloat16)
converted = loader.convert_state_dict(dict(sd))
print("input keys", len(sd), "converted pairs", len(converted) // 2)
print("sample converted", list(converted)[:4])
print("names", sorted({k.split(".lora_")[0] for k in converted})[:5])
print("extra (non-lora) keys", [k for k in sd if "lora_" not in k][:8])

"""Compare the LoRA delta magnitude with the base weights at two candidate scalings."""

import torch
from safetensors.torch import load_file

path = "/xs-fd-nas/workspace_zhaoxinrui/models/h3-ref2va-turbo/minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors"
sd = load_file(path)
pairs = sorted({k.removesuffix(".lora_A.default.weight") for k in sd if k.endswith(".lora_A.default.weight")})
print("pairs", len(pairs))

base = load_file(
    "/xs-fd-nas/workspace_zhaoxinrui/models/MiniMax-H3-Ref2VA-BF16/Ref2VA/transformer/model-00001-of-00013.safetensors"
)
print("base keys sample", list(base)[:3])

for name in pairs[:6]:
    a = sd[f"{name}.lora_A.default.weight"].float()
    b = sd[f"{name}.lora_B.default.weight"].float()
    delta = b @ a
    print(f"{name}: A{a.shape} B{b.shape} ||delta||={delta.norm():.4f} "
          f"||delta||*1.0={delta.norm():.4f} ||delta||*0.0625={delta.norm()*0.0625:.4f}")

# effective scale check: energy of the delta relative to a typical weight scale
a0 = sd[f"{pairs[0]}.lora_A.default.weight"].float()
b0 = sd[f"{pairs[0]}.lora_B.default.weight"].float()
d0 = b0 @ a0
print("per-element std of delta", d0.std().item(), "std*0.0625", (d0 * 0.0625).std().item())

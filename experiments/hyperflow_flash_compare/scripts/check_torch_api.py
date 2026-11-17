import torch
import torch.nn.attention as attn

print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("varlen module", hasattr(attn, "varlen"))
print("dynamo cfg", hasattr(torch._dynamo.config, "fail_on_recompile_limit_hit"))

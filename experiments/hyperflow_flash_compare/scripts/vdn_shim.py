"""Minimal ``torch.accelerator`` shim for the VDN-H3 env (torch 2.5.1, diffusers 0.42).

diffusers 0.42 calls the PyTorch 2.6 accelerator API. The VDN env's torch predates it, and
the env lives on a full root filesystem, so upgrading in place is not possible; this shim
maps the six methods diffusers uses onto ``torch.cuda``.
"""

import torch

if not hasattr(torch, "accelerator"):

    class _Accelerator:
        def is_available(self):
            return torch.cuda.is_available()

        def device_count(self):
            return torch.cuda.device_count()

        def current_accelerator(self, *args, **kwargs):
            if not torch.cuda.is_available():
                return None
            return torch.device("cuda", torch.cuda.current_device())

        def get_memory_info(self, device=None):
            free, total = torch.cuda.mem_get_info(device)
            return free, total

        def reset_peak_memory_stats(self, device=None):
            torch.cuda.reset_peak_memory_stats(device)

        def max_memory_allocated(self, device=None):
            return torch.cuda.max_memory_allocated(device)

    torch.accelerator = _Accelerator()

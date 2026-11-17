"""Patch the installed HyperFlow INT8 module for Triton 3.2 on this host.

The kernels are defined inside a factory, so Triton 3.2 resolves ``tl`` from the module
globals and fails with ``NameError: tl is not defined``. Mirroring the imports at module
level fixes resolution for every launch path (including the torchrun re-launch used by
``--gpus N``).
"""

from pathlib import Path

TARGET = Path("/dev/shm/hfenv/lib/python3.12/site-packages/hyperflow_h3_flash/int8.py")
MARKER = "# triton-globals-patch"
SNIPPET = (
    "\n"
    f"{MARKER}\n"
    "try:\n"
    "    import triton  # noqa: F401\n"
    "    import triton.language as tl  # noqa: F401\n"
    "except ImportError:\n"
    "    triton = None\n"
    "    tl = None\n"
)

source = TARGET.read_text(encoding="utf-8")
if MARKER in source:
    print("already patched")
else:
    anchor = "import torch\n"
    index = source.index(anchor) + len(anchor)
    TARGET.write_text(source[:index] + SNIPPET + source[index:], encoding="utf-8")
    print("patched", TARGET)

"""Run the HyperFlow Flash CLI with Triton globals repaired and an optional INT8 GEMM backend.

The INT8 kernels are defined inside a factory, so Triton 3.2 resolves ``tl`` from the
module globals and fails with ``NameError: tl is not defined``. Injecting the imports
into the module namespace before the kernels are built fixes that resolution.

``HF_INT8_GEMM`` selects the int8 GEMM backend (``auto``, ``triton``, ``cublas``, ``bf16``).
"""

import functools
import os
import sys

import triton
import triton.language as tl

import hyperflow_h3_flash.int8 as int8

int8.triton = triton
int8.tl = tl

from hyperflow_h3_flash import cli, pipeline

_backend = os.environ.get("HF_INT8_GEMM")
if _backend:
    _load = pipeline.load_pipeline

    @functools.wraps(_load)
    def _patched(*args, **kwargs):
        kwargs.setdefault("int8_gemm", _backend)
        return _load(*args, **kwargs)

    pipeline.load_pipeline = _patched
    cli.load_pipeline = _patched

if __name__ == "__main__":
    sys.exit(cli.main())

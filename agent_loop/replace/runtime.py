"""SAM adapter for the replacement workflow."""

import sys
import os
from pathlib import Path

from ..runtime import ROOT, dump, load, sha

SAM = Path(os.environ.get("CROSSVIEW_SAM_ROOT", ROOT / "deps/sam3"))


def sam_predictor():
    import inspect

    checkpoint = Path(os.environ.get("CROSSVIEW_SAM_CHECKPOINT", SAM / "sam3.1/sam3.1_multiplex.pt"))
    if not SAM.is_dir() or not checkpoint.is_file():
        raise FileNotFoundError("Prepare SAM with scripts/setup_sam.sh and set CROSSVIEW_SAM_CHECKPOINT")
    sys.path.insert(0, str(SAM))
    from sam3.model_builder import build_sam3_predictor

    model = build_sam3_predictor(checkpoint_path=str(checkpoint),
                                version='sam3.1', compile=False, use_fa3=False,
                                use_rope_real=False, async_loading_frames=False)
    original = model.model.init_state
    if 'offload_state_to_cpu' not in inspect.signature(original).parameters:
        def compat(*args, **kwargs):
            if kwargs.pop('offload_state_to_cpu', False):
                raise ValueError('SAM version does not support CPU state offloading')
            return original(*args, **kwargs)
        model.model.init_state = compat
    return model

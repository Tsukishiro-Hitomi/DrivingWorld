"""Shared path resolution for the S2 lane-change experiment scripts.

Every script here is runnable from any working directory::

    python experiments/lane_change_s2/scripts/eval_s2.py a1

Paths are derived from this file, so local checkouts and the dev box use the
same tree. Override the roots with environment variables when a run lives
outside the default layout:

    DRIVINGWORLD_ROOT         repository checkout
    DRIVINGWORLD_MODEL_CACHE  directory holding the downloaded checkpoints

The experiment scripts were originally `.tmp_*` scratch files run from the
repository root; they depended on hard-coded absolute paths to the dev box.
Keeping that out of the committed code is the main reason this module exists.
"""

from __future__ import annotations

import os
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
EXPERIMENT = SCRIPTS.parent
REPO = Path(os.environ.get("DRIVINGWORLD_ROOT", EXPERIMENT.parents[1])).resolve()

#: `experiments/lane_change_s2`; the scripts call it `S2` throughout.
S2 = EXPERIMENT
RESULTS = S2 / "results"

#: Generated, untracked intermediate files (contact sheets, schemas, crops).
SCRATCH = S2 / "scratch"

MODEL_CACHE = Path(
    os.environ.get(
        "DRIVINGWORLD_MODEL_CACHE", "/xs-fd-nas/workspace_zhaoxinrui/model_cache"
    )
)
QWEN3_VL_8B = MODEL_CACHE / "models/Qwen--Qwen3-VL-8B-Instruct"


def scratch_path(name: str) -> Path:
    """Return `S2/scratch/<name>`, creating the directory."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    return SCRATCH / name


def legacy_dir(name: str) -> Path:
    """Resolve a scratch directory, preferring the experiment scratch tree.

    Earlier runs wrote these at the repository root as `.tmp_<name>` and the
    dev box still has them there, so both locations are accepted.
    """
    preferred = SCRATCH / name
    if preferred.exists():
        return preferred
    return REPO / f".tmp_{name}"


def repo_script(*candidates: str) -> Path:
    """Return the first existing candidate, searched in scripts, scratch, repo root.

    Used for helpers that were never promoted out of the scratch tree.
    """
    for name in candidates:
        for base in (SCRIPTS, SCRATCH, REPO):
            path = base / name
            if path.exists():
                return path
    tried = ", ".join(str(REPO / name) for name in candidates)
    raise FileNotFoundError(f"none of these exist: {tried}")

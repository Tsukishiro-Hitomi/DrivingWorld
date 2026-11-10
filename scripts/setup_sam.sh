#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sam_root="${CROSSVIEW_SAM_ROOT:-$project_root/deps/sam3}"
sam_revision=8f0b7f4d4e7eda2ed606ebde6702c93359ad01da
sam_patch="$project_root/patches/sam3-retain-prompt-memory.patch"
if [ ! -d "$sam_root/.git" ]; then
    if [ -e "$sam_root" ]; then
        printf '%s\n' 'SAM destination already exists and is not a Git checkout.' >&2
        exit 1
    fi
    git clone --no-checkout https://github.com/facebookresearch/sam3.git "$sam_root"
    git -C "$sam_root" checkout --detach "$sam_revision"
fi
if [ "$(git -C "$sam_root" rev-parse HEAD)" != "$sam_revision" ]; then
    printf '%s\n' 'SAM revision differs; existing checkout was preserved.' >&2
    exit 1
fi
if ! git -C "$sam_root" apply --reverse --check "$sam_patch" 2>/dev/null; then
    git -C "$sam_root" apply --check "$sam_patch"
    git -C "$sam_root" apply "$sam_patch"
fi
"${PYTHON:-python}" -m pip install --no-deps -e "$sam_root"
printf '%s\n' 'SAM code ready. Supply the separately licensed checkpoint via CROSSVIEW_SAM_CHECKPOINT.'

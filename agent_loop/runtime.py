"""Shared file provenance for driving-edit agents."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def dump(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def load(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

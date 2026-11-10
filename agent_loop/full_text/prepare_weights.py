"""Fetch a pinned FL2VA checkpoint through a mirror and verify every file."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import threading
import urllib.request
import ssl

from ..runtime import ROOT, dump

REPO = "MiniMaxAI/MiniMax-H3"


def identity(path, info):
    if not path.is_file() or path.stat().st_size != info["size"]:
        return False
    if info.get("lfs"):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(part)
        return digest.hexdigest() == info["lfs"]["sha256"]
    digest = hashlib.sha1(f"blob {info['size']}\0".encode() + path.read_bytes()).hexdigest()
    return digest == info["blobId"]


def prepare(destination, workers=4):
    from huggingface_hub import hf_hub_download
    import certifi

    destination.mkdir(parents=True, exist_ok=True)
    endpoint = "https://hf-mirror.com"
    with urllib.request.urlopen(endpoint + "/api/models/" + REPO + "?blobs=true", timeout=60,
                                context=ssl.create_default_context(cafile=certifi.where())) as response:
        metadata = json.load(response)
    revision = metadata["sha"]
    files = [item for item in metadata["siblings"] if item["rfilename"].startswith("FL2VA/")]
    previous = {item["rfilename"][len("Ref2VA/"):]: item for item in metadata["siblings"]
                if item["rfilename"].startswith("Ref2VA/")}
    reference = ROOT / "models/MiniMax-H3-Ref2VA-BF16/Ref2VA"
    manifest = dict(repo=REPO, revision=revision, primary_endpoint=endpoint, files=files)
    dump(destination / "download_manifest.json", manifest)
    state = dict(stage="downloading", revision=revision, verified_files=0, verified_bytes=0,
                 reused_bytes=0, total_files=len(files), total_bytes=sum(f["size"] for f in files))
    lock = threading.Lock()
    dump(destination / "download_state.json", state)

    def reusable(info):
        key = info["rfilename"][len("FL2VA/"):]
        old = previous.get(key, {})
        return (bool(info.get("lfs")) and info["lfs"]["sha256"] == old.get("lfs", {}).get("sha256")
                and (reference / key).is_file())

    def fetch(info):
        name = info["rfilename"]
        target = destination / name
        used_existing = False
        if target.exists() and identity(target, info):
            used_existing = True
        elif reusable(info):
            source = reference / name[len("FL2VA/"):]
            if identity(source, info):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.unlink(missing_ok=True)
                os.link(source, target)
                used_existing = True
        if not used_existing:
            # Never let a download truncate a reused checkpoint hardlink.
            if target.exists() and target.stat().st_nlink > 1:
                target.unlink()
            try:
                hf_hub_download(REPO, name, revision=revision, local_dir=destination, endpoint=endpoint)
            except Exception as mirror_error:
                print("MIRROR_FAILED_FALLBACK", name, type(mirror_error).__name__, flush=True)
                hf_hub_download(REPO, name, revision=revision, local_dir=destination, endpoint="https://huggingface.co")
            if not identity(target, info):
                raise ValueError("Downloaded checkpoint file failed its Hub identity: " + name)
        with lock:
            state["verified_files"] += 1
            state["verified_bytes"] += info["size"]
            if used_existing:
                state["reused_bytes"] += info["size"]
            state["last_file"] = name
            dump(destination / "download_state.json", state)
            print("VERIFIED", state["verified_files"], len(files), name, "reused" if used_existing else "downloaded", flush=True)

    ordered = sorted(files, key=lambda f: (reusable(f), -f["size"]))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, info) for info in ordered]
        for future in as_completed(futures):
            future.result()
    state["stage"] = "complete"
    dump(destination / "download_state.json", state)
    (destination / ".ready").write_text(revision + "\n")
    print("FL2VA_READY", destination, json.dumps(state), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "models/MiniMax-H3-FL2VA-BF16")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    prepare(args.destination.resolve(), args.workers)


if __name__ == "__main__":
    main()

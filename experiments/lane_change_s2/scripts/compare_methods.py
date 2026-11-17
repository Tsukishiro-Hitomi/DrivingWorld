"""Compare every judge on the common subset that has a 1b filmstrip."""
import json
from pathlib import Path

from _paths import S2  # noqa: E402
RESULTS = S2 / "results"

METHODS = [
    ("geom A1 (GT track)", "a1.jsonl"),
    ("full-frame Qwen", "vlm.jsonl"),
    ("full-frame Luna", "luna.jsonl"),
    ("film Qwen", "film_qwen.jsonl"),
    ("film Luna", "film_luna.jsonl"),
    ("film1b Qwen", "film1b_qwen.jsonl"),
    ("film1b Luna", "film1b_luna.jsonl"),
]


def read(path):
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        v = r.get("pred")
        if isinstance(v, bool):
            v = int(v)
        elif isinstance(v, str):
            v = 1 if v.strip().lower() in ("yes", "true", "1", "positive") else 0
        elif v is not None:
            v = int(v)
        out[r["clip_id"]] = v
    return out


def main():
    labels = {}
    for line in (S2 / "labels.jsonl").open():
        if line.strip():
            r = json.loads(line)
            labels[r["clip_id"]] = r

    subset = {p.stem for p in (S2 / "sheets_film1b").glob("*.png")}
    pos = sum(1 for c in subset if labels[c]["label"] == 1)
    neg = sum(1 for c in subset if labels[c]["label"] == 0)
    print(f"common subset: {len(subset)} clips ({pos} pos / {neg} neg)\n")
    print(f"{'method':<22}{'recall':>16}{'fpr':>16}{'judged':>10}")
    for name, fname in METHODS:
        preds = read(RESULTS / fname)
        tp = n_pos = fp = n_neg = judged = 0
        for cid in subset:
            lab = labels[cid]
            p = preds.get(cid)
            if p is not None:
                judged += 1
            else:
                p = 0
            if lab["label"] == 1:
                n_pos += 1
                tp += p
            else:
                n_neg += 1
                fp += p
        print(f"{name:<22}{f'{tp}/{n_pos}={tp/max(n_pos,1):.3f}':>16}"
              f"{f'{fp}/{n_neg}={fp/max(n_neg,1):.3f}':>16}{judged:>10}")

    rep = RESULTS / "s2_dets_report.json"
    if rep.exists():
        data = json.loads(rep.read_text())
        by = {c["clip_id"]: c for c in data["clips"]}
        tp = n_pos = fp = n_neg = 0
        for cid in subset:
            lab = labels[cid]
            p = (by.get(cid) or {}).get("pred", 0)
            if lab["label"] == 1:
                n_pos += 1
                tp += p
            else:
                n_neg += 1
                fp += p
        print(f"{'detector + known size':<22}{f'{tp}/{n_pos}={tp/max(n_pos,1):.3f}':>16}"
              f"{f'{fp}/{n_neg}={fp/max(n_neg,1):.3f}':>16}")


if __name__ == "__main__":
    main()

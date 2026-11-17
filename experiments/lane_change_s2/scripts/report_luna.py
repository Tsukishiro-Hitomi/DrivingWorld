"""Validate and summarize the GPT-6-Luna PREVENTION S2 result."""

import json
from collections import Counter, defaultdict
from pathlib import Path

from _paths import REPO, S2  # noqa: E402
samples = json.loads((S2 / "samples.json").read_text())
rows = [json.loads(line) for line in (S2 / "results/luna.jsonl").read_text().splitlines() if line.strip()]
sample_by_id = {x["clip_id"]: x for x in samples}

assert len(samples) == 221
assert len(rows) == 221
assert len({x["clip_id"] for x in rows}) == 221
assert {x["clip_id"] for x in rows} == set(sample_by_id)
for row in rows:
    assert set(row) == {"clip_id", "pred", "shift", "raw"}
    assert row["pred"] in (0, 1, None)

by = defaultdict(list)
for row in rows:
    by[sample_by_id[row["clip_id"]]["stratum"]].append(row)

print("=== luna [pred] ===")
for stratum in ("cut_in", "cut_out", "lateral", "negative_clean", "negative_hard"):
    values = by[stratum]
    positives = sum(row["pred"] == 1 for row in values)
    nulls = sum(row["pred"] is None for row in values)
    metric = "recall" if sample_by_id[values[0]["clip_id"]]["label"] else "FPR"
    print(f"  {stratum:<16} {metric:<6} {positives}/{len(values)} = {positives / len(values):.4f}"
          f"  (null {nulls})")

positive_rows = [row for row in rows if sample_by_id[row["clip_id"]]["label"] == 1]
negative_rows = [row for row in rows if sample_by_id[row["clip_id"]]["label"] == 0]
hits = sum(row["pred"] == 1 for row in positive_rows)
fp = sum(row["pred"] == 1 for row in negative_rows)
print(f"  {'ALL positives':<16} recall {hits}/76 = {hits / 76:.4f}")
print(f"  {'ALL negatives':<16} FPR    {fp}/145 = {fp / 145:.4f}")
print(f"  coverage         {sum(row['pred'] is not None for row in rows)}/221; nulls={sum(row['pred'] is None for row in rows)}")

print("\nFalse positives:")
for row in negative_rows:
    if row["pred"] == 1:
        s = sample_by_id[row["clip_id"]]
        print(row["clip_id"], s["stratum"], f"depth={s['depth_m']}", f"direction={s['direction']}",
              f"shift={row['shift']}", row["raw"])

print("\nNulls:")
for row in rows:
    if row["pred"] is None:
        s = sample_by_id[row["clip_id"]]
        print(row["clip_id"], s["stratum"], f"depth={s['depth_m']}", f"direction={s['direction']}")

print("\nPositive breakdown by direction/depth:")
for direction in sorted({sample_by_id[row["clip_id"]]["direction"] for row in positive_rows}):
    group = [row for row in positive_rows if sample_by_id[row["clip_id"]]["direction"] == direction]
    print(direction, sum(row["pred"] == 1 for row in group), "/", len(group))
for lo, hi in ((0, 30), (30, 50), (50, 10_000)):
    group = [row for row in positive_rows if lo <= sample_by_id[row["clip_id"]]["depth_m"] < hi]
    print(f"depth[{lo},{hi})", sum(row["pred"] == 1 for row in group), "/", len(group))

print("\nPrediction/shift counts:", Counter((row["pred"], row["shift"]) for row in rows).most_common(20))

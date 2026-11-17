"""Does Luna's reported sign agree with the true lateral direction on positives?"""
import json
from pathlib import Path

from _paths import S2  # noqa: E402
samples = {s["clip_id"]: s for s in json.loads((S2 / "samples.json").read_text(encoding="utf-8"))}
labels = {json.loads(l)["clip_id"]: json.loads(l) for l in
          (S2 / "labels.jsonl").open() if l.strip()}
rows = [json.loads(l) for l in (S2 / "results/film1b_luna.jsonl").open() if l.strip()]

# camera frame is x right; a leftward move lowers x, so a left change is a negative shift
agree = disagree = zero = 0
per_dir = {}
for r in rows:
    lab = labels.get(r["clip_id"])
    smp = samples.get(r["clip_id"])
    if lab is None or smp is None or lab["label"] != 1:
        continue
    shift = r.get("shift")
    if shift is None:
        continue
    d = smp.get("direction")
    if d not in ("left", "right"):
        continue
    exp = -1 if d == "left" else 1
    got = 0 if shift == 0 else (1 if shift > 0 else -1)
    key = d
    per_dir.setdefault(key, [0, 0, 0])
    if got == 0:
        zero += 1
        per_dir[key][2] += 1
    elif got == exp:
        agree += 1
        per_dir[key][0] += 1
    else:
        disagree += 1
        per_dir[key][1] += 1

tot = agree + disagree
print(f"positives with a signed reading and a known direction: {tot}")
print(f"  sign agrees with truth: {agree} ({agree/tot:.3f})")
print(f"  sign flips:            {disagree} ({disagree/tot:.3f})")
print(f"  exact zero (no call):  {zero}")
for d, (a, b, z) in sorted(per_dir.items()):
    n = a + b
    print(f"  direction {d:<5}: agree {a}/{n} = {a/max(n,1):.3f}, zero {z}")

# magnitude split for positives that were called "yes"
yes = [r for r in rows if r.get("pred") == 1 and
       labels.get(r["clip_id"], {}).get("label") == 1]
if yes:
    mags = sorted(abs(float(r["shift"])) for r in yes)
    print(f"\npositives called yes: {len(yes)}, |SHIFT| median {mags[len(mags)//2]:.2f}, "
          f"max {mags[-1]:.2f}  (a real lane change is ~2 car widths)")

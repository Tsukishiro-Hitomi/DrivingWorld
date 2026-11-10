# Event Judge Dataset

Each record in `manifest.jsonl` pairs one generated H3 video with one requested event and a human annotation.
Videos are stored below `videos/`; model outputs belong in `reports/<model>/` and never modify the manifest.

The first collection pass must include successful events, partial events, hard negatives, and indeterminate cases. The annotation describes the H3 output, not the proxy trajectory that requested it.

`event` follows `driving_world.contracts.events.EventSpec`. `annotation` follows `EventVerdict`. Both carry the same `event_id`; the annotation includes evidence-frame indexes from the chosen review sample.

The empty manifest is intentional until generated videos receive human labels. Use the dataset builder in the next phase to add records.

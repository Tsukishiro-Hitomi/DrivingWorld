"""Shared offline Qwen JSON worker; GPU processes are released between stages."""

import json
import os
import subprocess
import sys

from .runtime import ROOT, dump

QWEN_PYTHON = os.environ.get(
    "CROSSVIEW_QWEN_PYTHON",
    str(ROOT / ".venv-qwen/bin/python") if (ROOT / ".venv-qwen/bin/python").is_file() else sys.executable,
)


def qwen_json(messages, out, role, max_tokens=1400):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from pathlib import Path

    weights = Path(os.environ.get("CROSSVIEW_QWEN_MODEL_ROOT", ROOT / "models/Qwen3.8-27B"))
    processor = AutoProcessor.from_pretrained(weights, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        weights, local_files_only=True, device_map={"": "cuda:0"}, dtype=torch.bfloat16
    ).eval()
    for attempt in range(2):
        batch = processor.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False,
            tokenize=True, return_dict=True, return_tensors="pt"
        ).to("cuda:0")
        with torch.inference_mode():
            result = model.generate(**batch, do_sample=False, max_new_tokens=max_tokens)
        raw = processor.decode(result[0, batch.input_ids.shape[1]:], skip_special_tokens=True).strip()
        dump(out / f"{role}_response_{attempt}.json", {"text": raw})
        try:
            answer = json.loads(raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip())
            if not isinstance(answer, dict):
                raise ValueError("Expected a JSON object")
            return answer
        except ValueError:
            if attempt:
                raise
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "Return a single valid JSON object only."},
            ]



def run_worker(module, action, out, log_name):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="0", HF_ENABLE_PARALLEL_LOADING="true",
               HF_PARALLEL_LOADING_WORKERS="6")
    with (out / log_name).open("w") as log:
        subprocess.run([QWEN_PYTHON, "-m", module, action, "--out", str(out)],
                       cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=1800)

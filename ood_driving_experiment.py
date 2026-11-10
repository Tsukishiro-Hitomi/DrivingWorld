"""Run the three-way Ref2VA prompt ablation against verified archived baselines."""

import argparse
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
import time

from agent_loop.runtime import dump, load, sha

ROOT = Path(__file__).resolve().parent
SCENES = load(ROOT / "configs/ood_driving_prompt.json")
SKILL = ROOT / "skills/ood-driving-prompt/SKILL.md"
MODEL_ID = "Qwen/Qwen3.8-27B"
FPS, WIDTH, HEIGHT, SEED, STEPS = 24, 832, 480, 20261028, 8
ARMS = ("baseline", "baseline-LLM", "baseline-LLM-skill")
FIELDS = ("subject_definitions", "summary", "retention_analysis", "detailed_description",
          "overall_soundscape", "non_diegetic_music")
LABEL = re.compile(r"<(Picture|Video|Audio|Subject) (\d+)>")


def folder(name, arm):
    return ROOT / "outputs" / f"{name}-H3-{arm}"


def probe(video, count):
    import cv2

    cap = cv2.VideoCapture(str(video))
    frames = 0
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape != (HEIGHT, WIDTH, 3):
                raise ValueError(f"Unexpected frame dimensions in {video}: {frame.shape}")
            frames += 1
    finally:
        cap.release()
    if frames != count or abs(fps - FPS) > .001:
        raise ValueError(f"{video}: expected {count}/{FPS}, decoded {frames}/{fps}")


def checked(path, expected):
    value = sha(path)
    if value != expected:
        raise ValueError(f"Archived asset mismatch: {path}")
    return value


def baseline(name, spec):
    video = ROOT / spec["baseline"]
    if "image" in spec:
        prompt = spec["baseline_prompt"]
        hashes = {
            "image_sha256": checked(ROOT / spec["image"], spec["image_sha256"]),
            "video_sha256": checked(video, spec["baseline_sha256"]),
        }
    else:
        archived = load(ROOT / spec["baseline_metadata"])
        prompt = archived["prompt"]
        hashes = {
            f"{key}_sha256": checked(ROOT / spec[key], archived[f"{key}_sha256"])
            for key in ("source", "warp")
        }
        hashes["video_sha256"] = checked(video, archived["generated_sha256"])
        probe(ROOT / spec["source"], spec["frames"])
        probe(ROOT / spec["warp"], spec["frames"])
    probe(video, spec["frames"])
    target = folder(name, ARMS[0])
    target.mkdir(parents=True, exist_ok=True)
    dest = target / video.name
    if not dest.exists():
        with TemporaryDirectory(dir=target) as staging:
            copy = Path(staging) / video.name
            shutil.copyfile(video, copy)
            copy.replace(dest)
    checked(dest, hashes["video_sha256"])
    dump(target / "experiment.json", {
        "scene": name, "arm": ARMS[0], "reused": True, "video": dest.name,
        "archived_inputs": spec, "prompt": prompt, **settings(spec), **hashes,
    })
    print(f"Verified baseline {name}: {dest.name}", flush=True)


def settings(spec):
    return {
        "h3_model": "MiniMax-H3 Ref2VA BF16", "frames": spec["frames"], "fps": FPS,
        "width": WIDTH, "height": HEIGHT, "seed": SEED, "steps": STEPS,
        "video_shift": 12, "audio_shift": 3, "crossview_strength": .8,
        "image_condition_noise_aug": .999, "acceleration": "Larry v4 step600 EMA",
        "reference_image_short_edge": 1024 if "image" in spec else None,
        "cuda_visible_devices": spec["gpus"],
        "guide": "aligned Warp, all 8 steps" if "warp" in spec else None,
        "first_frame_anchor": "source first frame" if "source" in spec else None,
    }


def messages(spec, archived, skill):
    reference = ("one ordinary scene image (<Picture 1>), no exact keyframe, video or Warp"
                 if "image" in spec else
                 "one silent source video (<Video 1>) and a separate aligned CrossView Warp guide")
    system = (
        "Rewrite a driving scenario brief into a standalone English video-generation prompt. "
        "Return the prompt body only, without Markdown wrappers, commentary or LoRA trigger. "
        "Preserve the event, direction, clip duration and asset roles. "
        "The exported clip is silent. Do not add dialogue, music, visible text or new events."
    )
    if skill:
        headings = "\n".join(f"{field}:" for field in FIELDS)
        system += (
            f"\n\nFollow this prompt-writing skill:\n{skill}\n\n"
            f"Write these six unadorned headings, including each trailing colon:\n{headings}\n"
            "Write N/A as the entire value of each of the last two sections."
        )
        if "ref2va_note" in spec:
            system += f"\n{spec['ref2va_note']}"
    else:
        system += " Use plain English prose in a single paragraph."
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": (
            f"Mode: Ref2VA\nDuration: {spec['frames'] / FPS:.2f} seconds\n"
            f"Inputs: {reference}\nScenario: {spec['intent']}\n"
            f"Existing baseline prompt (for context): {archived}"
        )},
    ]


def normalize_ref2va(body, spec):
    # Canonicalize header punctuation and task-type casing without changing the prose.
    headers = "|".join(FIELDS)
    body = re.sub(rf"^({headers})[ \t]*:?[ \t]*(?=\n|$)", r"\1:", body, flags=re.M)
    body = re.sub(r"\[(video editing|reference generation)\]", lambda m: m[0].lower(),
                  body, flags=re.I)
    # A shot marker is formatting. Insert it at the first shot boundary if omitted.
    detailed = re.search(r"(?ms)^detailed_description:[ \t]*(.*?)^overall_soundscape:", body)
    if detailed and not re.search(r"\[shot\s", detailed[1], re.I):
        shot = detailed[1].lstrip()
        opening = re.match(r".+?[.!?](?=\s)", shot)
        if opening and shot[opening.end():].strip():
            start = detailed.start(1) + len(detailed[1]) - len(shot) + opening.end()
            body = body[:start] + "\n[Shot 1] " + body[start:].lstrip()
    if "image" in spec:
        body = retain_image_subjects(body)
    return body


def retain_image_subjects(body):
    """Keep only subject labels grounded in the single supplied scene image."""
    sections = re.search(
        r"(?ms)^subject_definitions:[ \t]*(.*?)^summary:(.*?)^retention_analysis:(.*?)^detailed_description:",
        body,
    )
    if not sections:
        return body
    definitions = sections[1].strip().splitlines()
    retention = sections[3].strip().splitlines()
    subject = re.compile(r"^\s*(<Subject \d+>)")
    ungrounded = {m[1] for line in definitions if "<Picture 1>" not in line
                  and (m := subject.match(line))}
    if not ungrounded:
        return body
    # Never remove a subject name from the timeline or summary.
    if any(label in sections[2] or label in body[sections.end():] for label in ungrounded):
        return body
    defs = "\n".join(line for line in definitions
                      if not any(label in line for label in ungrounded))
    kept = "\n".join(line for line in retention
                      if not any(label in line for label in ungrounded))
    updated = (body[:sections.start(1)] + "\n" + defs + "\n"
               + body[sections.end(1):sections.start(3)] + "\n" + kept + "\n"
               + body[sections.end(3):])
    labels = dict.fromkeys(re.findall(r"<Subject \d+>", defs))
    mapping = {label: f"<Subject {i}>" for i, label in enumerate(labels, 1)}
    return re.sub(r"<Subject \d+>", lambda m: mapping.get(m[0], m[0]), updated)


def validate(body, spec, structured):
    if not body or "```" in body or "<think>" in body or re.match(r"crossview\b", body, re.I):
        raise ValueError("Invalid prompt body or duplicate LoRA trigger")
    headers = list(re.finditer(r"^([a-z_]+):", body, re.M))
    if not structured:
        if headers or "\n\n" in body or "[Shot " in body:
            raise ValueError("Plain ablation must be one paragraph")
        return
    if [header[1] for header in headers] != list(FIELDS) or headers[0].start() != 0:
        raise ValueError("Ref2VA prompt must use the six official fields in order")
    sections = {
        header[1]: body[header.end():end].strip()
        for header, end in zip(headers, [h.start() for h in headers[1:]] + [len(body)])
    }
    if any(not value for value in sections.values()):
        raise ValueError("Ref2VA fields cannot be empty")
    if re.findall(r"\[Shot (\d+)\]", sections["detailed_description"]) != ["1"]:
        raise ValueError("Expected a single uninterrupted shot")
    image = "image" in spec
    expected = ("Picture" if image else "Video", "1")
    used = set(LABEL.findall(body))
    defined = set(LABEL.findall(sections["subject_definitions"]))
    if expected not in defined or used - defined or {key for key in used if key[0] != "Subject"} != {expected}:
        raise ValueError("Prompt refers to unavailable or undefined media")
    if image and any(LABEL.match(line.strip()) and line.strip().startswith("<Subject ")
                     and "<Picture 1>" not in line for line in sections["subject_definitions"].splitlines()):
        raise ValueError("Ref2VA subject definitions must cite the supplied scene image")
    prefix = "[reference generation]" if image else "[video editing] The target video is an edited version of <Video 1>."
    if not sections["summary"].startswith(prefix):
        raise ValueError("Incorrect Ref2VA task type or source declaration")
    if any(sections[key] != "N/A" for key in FIELDS[-2:]):
        raise ValueError("This ablation exports silent clips")


def rewrite(scenes):
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    skill = SKILL.read_text(encoding="utf-8").split("---", 2)[2].strip()
    pending = []
    for name in scenes:
        spec = SCENES[name]
        verified = load(folder(name, ARMS[0]) / "experiment.json")
        archived = verified["prompt"]
        keys = ("image",) if "image" in spec else ("source", "warp")
        hashes = {f"{key}_sha256": verified[f"{key}_sha256"] for key in keys}
        for arm in ARMS[1:]:
            structured = arm == ARMS[2]
            target = folder(name, arm)
            target.mkdir(parents=True, exist_ok=True)
            inputs = messages(spec, archived, skill if structured else "")
            request = {
                "scene": name, "arm": arm, "llm": MODEL_ID, "llm_dtype": "bfloat16",
                "llm_source": "ModelScope", "llm_decoding": "greedy, thinking disabled",
                "skill_sha256": sha(SKILL) if structured else None,
                "llm_messages": inputs, **settings(spec), **hashes,
            }
            if (target / "prompt.json").exists():
                saved = load(target / "prompt.json")
                if any(saved[k] != value for k, value in request.items()):
                    raise ValueError(f"Saved prompt inputs differ: {target}")
                raw = saved["llm_output"]
                body = normalize_ref2va(raw, spec) if structured else raw
                validate(body, spec, structured)
                prompt = f"crossview.\n{body}" if spec["trigger"] else body
                if prompt != saved["prompt"]:
                    if (target / "generated.mp4").exists():
                        raise ValueError(f"Saved video used an earlier prompt: {target}")
                    dump(target / "prompt.json", saved | {"formatted_output": body, "prompt": prompt})
                    print(f"Updated Ref2VA formatting {target.name}", flush=True)
            else:
                pending.append((target, spec, structured, request))

    if not pending:
        return
    weights = ROOT / "models/Qwen3.8-27B"
    print(f"Loading local {MODEL_ID} from {weights}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(weights, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        weights, local_files_only=True, device_map={"": "cuda:0"}, dtype=torch.bfloat16,
    ).eval()
    for target, spec, structured, request in pending:
        inputs = request["llm_messages"]
        for attempt in range(2):
            batch = tokenizer.apply_chat_template(
                inputs, add_generation_prompt=True, enable_thinking=False,
                tokenize=True, return_dict=True, return_tensors="pt",
            ).to("cuda:0")
            with torch.inference_mode():
                output = model.generate(**batch, max_new_tokens=1200, do_sample=False)
            raw = tokenizer.decode(output[0, batch.input_ids.shape[1]:], skip_special_tokens=True).strip()
            body = normalize_ref2va(raw, spec) if structured else raw
            try:
                validate(body, spec, structured)
            except ValueError as error:
                if attempt:
                    raise ValueError(f"{target.name}: {error}\n{body}") from error
                inputs = [*inputs, {"role": "assistant", "content": body},
                          {"role": "user", "content": f"Regenerate the complete prompt. Fix: {error}"}]
            else:
                prompt = f"crossview.\n{body}" if spec["trigger"] else body
                dump(target / "prompt.json", request | {
                    "llm_output": raw, "formatted_output": body, "prompt": prompt, "format_retry": attempt,
                    "effective_llm_messages": inputs,
                })
                print(f"Saved Qwen rewrite {target.name}", flush=True)
                break


def generate(name, arm):
    import torch
    import crossview_v2v as h3

    spec = SCENES[name]
    target = folder(name, arm)
    request = load(target / "prompt.json")
    prompt = request["prompt"]
    expected_hashes = ("image",) if "image" in spec else ("source", "warp")
    for key in expected_hashes:
        checked(ROOT / spec[key], request[f"{key}_sha256"])
    video = target / "generated.mp4"
    if video.exists():
        record = load(target / "experiment.json")
        if record["prompt"] != prompt:
            raise ValueError(f"Refusing to replace existing result: {video}")
        checked(video, record["video_sha256"])
        probe(video, spec["frames"])
        return
    devices = [f"cuda:{i}" for i in range(len(spec["gpus"].split(",")))]
    if torch.cuda.device_count() != len(devices):
        raise ValueError(f"Launch with CUDA_VISIBLE_DEVICES={spec['gpus']}")

    print(f"Generating {target.name}", flush=True)
    start = time.monotonic()
    (ROOT / "logs").mkdir(exist_ok=True)
    with TemporaryDirectory(prefix=f"{name}-", dir=ROOT / "logs") as staging:
        work = Path(staging)
        if "image" in spec:
            infer_image(ROOT / spec["image"], prompt, work / video.name,
                        SEED, spec["frames"], devices)
        else:
            for key in expected_hashes:
                (work / f"{key}.mp4").symlink_to(ROOT / spec[key])
            dump(work / "config.json", settings(spec))
            h3.OUT, h3.PROMPT, h3.SEED, h3.DEVICES = work, prompt, SEED, devices
            h3.infer()
        result = work / video.name
        probe(result, spec["frames"])
        runtime = load(work / "generation.json") if "source" in spec else None
        dump(target / "experiment.json", {
            "scene": name, "arm": arm, "reused": False, "video": video.name,
            "archived_inputs": {key: spec[key] for key in expected_hashes},
            "prompt": prompt, **settings(spec), **{f"{key}_sha256": request[f"{key}_sha256"] for key in expected_hashes},
            "video_sha256": sha(result), "runtime": runtime,
            "wall_seconds": round(time.monotonic() - start, 3),
        })
        result.replace(video)
    print(f"Verified H3 result {target.name}", flush=True)


def infer_image(reference, prompt, output, seed, frames, devices):
    """Replay the original image-reference baseline path."""
    import gc

    from PIL import Image
    import torch

    from crossview_v2v import MODEL, WEIGHTS, attach_crossview
    from diffsynth.pipelines.minimax_h3_audio_video import (
        MiniMaxH3Unit_NoiseInitializer,
        MiniMaxH3Unit_PackedSequenceBuilder,
        MiniMaxH3Unit_PromptEmbedder,
        MiniMaxH3Unit_ReferenceEncoder,
    )
    from h3_trajectory.acceleration import attach_larry
    from h3_trajectory.cli import write_video
    from h3_trajectory.placement import place_model
    from h3_trajectory.runtime import load_pipe, move_context, velocity

    height, width = HEIGHT, WIDTH

    with torch.inference_mode():
        print(f"Encoding one scene image; frames={frames}, fps={FPS}, seed={seed}", flush=True)
        vae = load_pipe("Ref2VA", ("video_vae",), devices[0], MODEL)
        vae.imgvid_cond_noise_aug = 0.999
        noise = MiniMaxH3Unit_NoiseInitializer().process(vae, seed, frames, height, width, "cpu")
        with Image.open(reference) as image:
            references = [{"type": "image", "image": image.convert("RGB")}]
        condition = MiniMaxH3Unit_ReferenceEncoder().process(
            vae, references, seed, noise["video_latents"], frames, 1024, 480, width * height
        )
        condition = move_context(condition, "cpu")
        blocks = condition.pop("ref_blocks")
        assert [block["kind"] for block in blocks] == ["image"]
        assert condition["ref_audio_anchor"] is None
        del vae, references
        gc.collect()
        torch.cuda.empty_cache()

        print("Encoding prompt", flush=True)
        text = load_pipe("Ref2VA", ("text_encoder",), devices[0], MODEL)
        prompt_data = MiniMaxH3Unit_PromptEmbedder().process(
            text, prompt, ref_blocks=blocks, height=height, width=width
        )
        packed = MiniMaxH3Unit_PackedSequenceBuilder().process(
            text, **prompt_data, **noise, ref_blocks=blocks
        )
        context = move_context(
            dict(condition, **packed, prompt_embeds=prompt_data["prompt_embeds"],
                 imgvid_cond_noise_aug=0.999, audio_cond_noise_aug=1.0),
            "cpu",
        )
        del text, blocks, prompt_data, packed, condition
        gc.collect()
        torch.cuda.empty_cache()

        print("Loading H3 DiT, CrossView Warp LoRA (0.8), and Larry LoRA", flush=True)
        pipe = load_pipe("Ref2VA", ("dit",), "cpu", MODEL)
        attach_crossview(pipe.dit)
        attach_larry(pipe.dit, WEIGHTS / "minimax_h3_turbo_v4_step600_ema.safetensors")
        place_model(pipe.dit, devices, mlp_chunk_tokens=2048)
        pipe.device = torch.device(devices[0])
        pipe.scheduler.set_timesteps(STEPS, shift=12.0)
        pipe.scheduler_audio.set_timesteps(STEPS, shift=3.0)
        video = noise["video_latents"].to(devices[0])
        audio = noise["audio_latents"].to(devices[0])
        context = move_context(context, devices[0])
        print("Sampling with one image reference and zero Warp guide tokens", flush=True)
        for i in range(STEPS):
            v, a = velocity(pipe, context, video, audio, float(pipe.scheduler.sigmas[i]),
                            float(pipe.scheduler_audio.sigmas[i]))
            video = pipe.scheduler.step(v, pipe.scheduler.timesteps[i], video)
            audio = pipe.scheduler_audio.step(a, pipe.scheduler_audio.timesteps[i], audio)
            print(f"Sampling {i + 1}/{STEPS}", flush=True)
        assert torch.isfinite(video).all()
        del pipe, context, audio, v, a
        gc.collect()
        torch.cuda.empty_cache()

        print("Decoding", flush=True)
        vae = load_pipe("Ref2VA", ("video_vae",), devices[0], MODEL)
        decoded = vae.video_vae.decode_video(
            video, dtype=torch.bfloat16, tiled=True, tile_size=256, tile_overlap=64
        ).cpu()
        write_video(decoded, output, frames, FPS)
        print("Complete", output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("baseline", "Verify and reuse archived videos"),
        ("rewrite", "Run local Qwen3.8-27B on the prompt requests"),
    ):
        selected = commands.add_parser(command, help=help_text)
        selected.add_argument("scene", nargs="?", choices=SCENES)
    run = commands.add_parser("generate", help="Run one H3 generation with its baseline inputs")
    run.add_argument("scene", choices=SCENES)
    run.add_argument("arm", choices=ARMS[1:])
    args = parser.parse_args()
    if args.command == "generate":
        generate(args.scene, args.arm)
        return
    scenes = [args.scene] if args.scene else SCENES
    if args.command == "baseline":
        for name in scenes:
            baseline(name, SCENES[name])
    else:
        rewrite(scenes)


if __name__ == "__main__":
    main()

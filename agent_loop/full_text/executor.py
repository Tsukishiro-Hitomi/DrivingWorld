"""Native FL2VA sampling with text only and an audited zero-condition token layout."""

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from ..runtime import ROOT, dump, load, sha
from ..video import read_frames
from .contracts import compile_prompt, validate, verify_origin

MODEL_ROOT = Path(os.environ.get("CROSSVIEW_FL2VA_MODEL_ROOT", ROOT / "models/MiniMax-H3-FL2VA-BF16"))


def checkpoint_ready(root=MODEL_ROOT):
    return ((root / ".ready").is_file() and (root / "download_manifest.json").is_file()
            and load(root / "download_state.json").get("stage") == "complete")


def _component_files(root, component):
    folder = root / "FL2VA" / {"dit": "transformer", "text_encoder": "text_encoder", "video_vae": "video_vae/source"}[component]
    if component == "video_vae":
        paths = [folder / "model.safetensors"]
    else:
        names = sorted(set(load(folder / "model.safetensors.index.json")["weight_map"].values()))
        if not names or any(Path(name).name != name for name in names):
            raise ValueError("Invalid FL2VA shard index")
        paths = [folder / name for name in names]
    if any(not path.is_file() for path in paths):
        raise ValueError("FL2VA checkpoint component is incomplete: " + component)
    return [str(path) for path in paths]


def _load(component, device, root):
    import torch
    from transformers import AutoProcessor
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig

    paths = _component_files(root, component)
    pipe = MiniMaxH3Pipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device,
        model_configs=[ModelConfig(path=paths if len(paths) > 1 else paths[0], skip_download=True)],
        processor_config=None, redirect_common_files=False)
    if getattr(pipe, component, None) is None:
        raise ValueError("FL2VA loader did not populate " + component)
    getattr(pipe, component).requires_grad_(False).eval()
    pipe.trajectory_variant = "FL2VA"
    if component == "text_encoder":
        pipe.processor = AutoProcessor.from_pretrained(str(root / "FL2VA/processor"), local_files_only=True)
        pipe.tokenizer = pipe.processor.tokenizer
    return pipe


def generate(out):
    request, planned = load(out / "request.json"), load(out / "plan.json")
    validate(planned)
    verify_origin(out, planned)
    if planned["status"] != "ready":
        raise ValueError("Full-text plan is not ready")
    if not checkpoint_ready():
        raise ValueError("FL2VA weights have not completed verified deployment")
    manifest = load(MODEL_ROOT / "download_manifest.json")
    if (MODEL_ROOT / ".ready").read_text().strip() != manifest["revision"]:
        raise ValueError("FL2VA deployment revision mismatch")
    config = {key: request[key] for key in ("frames", "fps", "width", "height", "seed", "steps")}
    config.update(agent="full-text", input_mode="T2VA", checkpoint_variant="FL2VA", model_root=str(MODEL_ROOT),
        model_revision=manifest["revision"], model_manifest_sha256=sha(MODEL_ROOT / "download_manifest.json"),
        plan_sha256=sha(out / "plan.json"), prompt=compile_prompt(planned), source=None,
        reference_video_count=0, reference_image_count=0, keyframe_count=0, aligned_warp=None,
        first_frame_anchor=None, video_shift=12, audio_shift=3, acceleration=None, crossview_strength=0,
        output_audio=False, max_retries=0)
    if (out / "config.json").exists() and load(out / "config.json") != config:
        raise ValueError("Full-text generation inputs changed")
    dump(out / "config.json", config)
    (out / "prompt.txt").write_text(config["prompt"] + "\n")
    marker = out / "generation_attempt.json"
    if marker.exists() and not (out / "generated.mp4").exists():
        raise ValueError("Previous full-text generation is incomplete; no automatic retry")
    if not marker.exists():
        dump(marker, dict(config_sha256=sha(out / "config.json"), max_retries=0))
    if load(marker)["config_sha256"] != sha(out / "config.json"):
        raise ValueError("Generation marker differs from full-text inputs")
    with (out / "h3.log").open("a") as log:
        subprocess.run([sys.executable, "-m", "agent_loop.full_text.executor", "infer", "--out", str(out)],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def infer(out):
    config = load(out / "config.json")
    if (config["input_mode"] != "T2VA" or config["checkpoint_variant"] != "FL2VA" or config["source"] is not None
            or any(config[key] != 0 for key in ("reference_video_count", "reference_image_count", "keyframe_count"))
            or config["aligned_warp"] is not None or config["first_frame_anchor"] is not None):
        raise ValueError("Full-text inference must have zero visual conditions")
    if (out / "generated.mp4").exists():
        generation = load(out / "generation.json")
        if generation["generated_sha256"] != sha(out / "generated.mp4") or any(generation.get(k) != v for k, v in config.items()):
            raise ValueError("Existing full-text output differs from its manifest")
        return
    root = Path(config["model_root"])
    if sha(root / "download_manifest.json") != config["model_manifest_sha256"]:
        raise ValueError("FL2VA model manifest changed")
    import torch
    from diffsynth.pipelines.minimax_h3_audio_video import (
        MiniMaxH3Unit_NoiseInitializer, MiniMaxH3Unit_PromptEmbedder, MiniMaxH3Unit_PackedSequenceBuilder,
    )
    from h3_trajectory.cli import write_video
    from h3_trajectory.placement import place_model
    from h3_trajectory.runtime import move_context, velocity

    with torch.inference_mode():
        print("Encoding text through native T2VA presentation; zero input media", flush=True)
        text_pipe = _load("text_encoder", "cuda:0", root)
        noise = MiniMaxH3Unit_NoiseInitializer().process(text_pipe, config["seed"], config["frames"],
                                                       config["height"], config["width"], "cpu")
        prompt = MiniMaxH3Unit_PromptEmbedder().process(text_pipe, config["prompt"], keyframes=None,
                                                     ref_blocks=None, height=config["height"], width=config["width"])
        packed = MiniMaxH3Unit_PackedSequenceBuilder().process(text_pipe, **prompt, **noise,
                                                             ref_blocks=None, keyframe_cond_anchor=None, keyframe_indices=None)
        layout = packed["packed"]
        shape = noise["video_latents"].shape
        video_tokens = shape[2] * (shape[3] // 2) * (shape[4] // 2)
        condition_tokens = int(layout["img_pos"].numel()) - video_tokens
        if condition_tokens != 0:
            raise ValueError("Full-text packed layout contains unexpected visual condition rows")
        dump(out / "input_audit.json", dict(input_mode="T2VA", checkpoint_variant="FL2VA", source_video=None,
            reference_video_count=0, reference_image_count=0, keyframe_count=0, visual_condition_tokens=condition_tokens,
            text_tokens=int(prompt["prompt_embeds"].shape[0]), video_tokens=video_tokens,
            video_latent_shape=list(shape), audio_latent_shape=list(noise["audio_latents"].shape),
            text_presentation="presentation_t2va", packed_layout="FL2VA with zero keyframes",
            crossview_adapter=False, acceleration_adapter=False))
        context = move_context(dict(**packed, prompt_embeds=prompt["prompt_embeds"],
            imgvid_cond_noise_aug=.999, audio_cond_noise_aug=1.), "cpu")
        noise = move_context(noise, "cpu")
        del text_pipe, prompt, packed, layout
        gc.collect()
        torch.cuda.empty_cache()

        print("Loading native FL2VA DiT", flush=True)
        pipe = _load("dit", "cpu", root)
        place_model(pipe.dit, ["cuda:0", "cuda:1"], mlp_chunk_tokens=2048)
        pipe.device = torch.device("cuda:0")
        pipe.scheduler.set_timesteps(config["steps"], shift=config["video_shift"])
        pipe.scheduler_audio.set_timesteps(config["steps"], shift=config["audio_shift"])
        video, audio = noise["video_latents"].to("cuda:0"), noise["audio_latents"].to("cuda:0")
        context = move_context(context, "cuda:0")
        for index in range(config["steps"]):
            v, a = velocity(pipe, context, video, audio, float(pipe.scheduler.sigmas[index]),
                            float(pipe.scheduler_audio.sigmas[index]))
            video = pipe.scheduler.step(v, pipe.scheduler.timesteps[index], video)
            audio = pipe.scheduler_audio.step(a, pipe.scheduler_audio.timesteps[index], audio)
            if index == 0 or (index + 1) % 5 == 0:
                dump(out / "run_state.json", dict(stage="sampling", step=index + 1, steps=config["steps"]))
                print("Sampling", index + 1, "/", config["steps"], flush=True)
        if not torch.isfinite(video).all():
            raise FloatingPointError("Nonfinite full-text video latent")
        video = video.cpu()
        del pipe, context, noise, audio, v, a
        gc.collect()
        torch.cuda.empty_cache()

        print("Decoding full-text video", flush=True)
        vae = _load("video_vae", "cuda:0", root)
        decoded = vae.video_vae.decode_video(video.to("cuda:0"), dtype=torch.bfloat16,
                                            tiled=True, tile_size=256, tile_overlap=64).cpu()
        with TemporaryDirectory(prefix="full-text-video-", dir=out) as temporary:
            path = Path(temporary) / "generated.mp4"
            write_video(decoded, path, config["frames"], config["fps"])
            read_frames(path)
            dump(out / "generation.json", dict(config, generated_sha256=sha(path),
                sampler="native FL2VA flow Euler", audio_exported=False, input_audit_sha256=sha(out / "input_audit.json")))
            path.replace(out / "generated.mp4")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("infer",))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    infer(args.out.resolve())


if __name__ == "__main__":
    main()

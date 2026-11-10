"""Shared source-video-plus-text Ref2VA inference, without image/warp anchors."""

import gc
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from .runtime import dump, load, sha
from .video import read_frames


def infer(out, config_name="config.json"):
    config = load(out / config_name)
    if (config.get("reference_image_count", 0) != 0 or config.get("aligned_warp") is not None
            or config.get("first_frame_anchor") is not None):
        raise ValueError("Shared source+text inference accepts no image, Warp or first-frame anchor")
    if "source" in config and sha(Path(config["source"])) != config["source_sha256"]:
        raise ValueError("Source video differs from the inference manifest")
    if (out / "generated.mp4").exists():
        generation = load(out / "generation.json")
        if (sha(out / "generated.mp4") != generation["generated_sha256"]
                or any(generation.get(k) != v for k, v in config.items())):
            raise ValueError("Existing generation does not match its input manifest")
        return
    FRAMES, FPS = config["frames"], config["fps"]
    HEIGHT, WIDTH = config.get("height", 480), config.get("width", 832)
    SEED = config["seed"]
    import torch
    from crossview_v2v import MODEL, WEIGHTS, attach_crossview
    from diffsynth.pipelines.minimax_h3_audio_video import (
        MiniMaxH3Unit_NoiseInitializer, MiniMaxH3Unit_ReferenceEncoder,
        MiniMaxH3Unit_PromptEmbedder, MiniMaxH3Unit_PackedSequenceBuilder,
    )
    from h3_trajectory.acceleration import attach_larry
    from h3_trajectory.cli import write_video
    from h3_trajectory.placement import place_model
    from h3_trajectory.runtime import load_pipe, move_context, velocity

    with torch.inference_mode():
        print("Encoding original video reference, with zero image or Warp inputs", flush=True)
        vae = load_pipe("Ref2VA", ("video_vae",), "cuda:0", MODEL)
        vae.imgvid_cond_noise_aug = .999
        noise = MiniMaxH3Unit_NoiseInitializer().process(vae, SEED, FRAMES, HEIGHT, WIDTH, "cpu")
        references = [{"type": "video", "video": [
            Image.fromarray(frame) for frame in read_frames(config["source"], FRAMES, FPS, WIDTH, HEIGHT)
        ]}]
        condition = MiniMaxH3Unit_ReferenceEncoder().process(
            vae, references, SEED, noise["video_latents"], FRAMES, 1024, HEIGHT, WIDTH * HEIGHT
        )
        condition = move_context(condition, "cpu")
        blocks = condition.pop("ref_blocks")
        if [block["kind"] for block in blocks] != ["video"]:
            raise ValueError("Text-only run must contain exactly one video reference")
        del vae, references
        gc.collect()
        torch.cuda.empty_cache()

        print("Encoding edit text", flush=True)
        text = load_pipe("Ref2VA", ("text_encoder",), "cuda:0", MODEL)
        prompt_data = MiniMaxH3Unit_PromptEmbedder().process(
            text, config["prompt"], ref_blocks=blocks, height=HEIGHT, width=WIDTH
        )
        packed = MiniMaxH3Unit_PackedSequenceBuilder().process(
            text, **prompt_data, **noise, ref_blocks=blocks
        )
        context = move_context(
            dict(condition, **packed, prompt_embeds=prompt_data["prompt_embeds"],
                 imgvid_cond_noise_aug=.999, audio_cond_noise_aug=1.),
            "cpu",
        )
        del text, blocks, prompt_data, packed, condition
        gc.collect()
        torch.cuda.empty_cache()

        print("Loading H3 with CrossView 0.8 and Larry v4", flush=True)
        pipe = load_pipe("Ref2VA", ("dit",), "cpu", MODEL)
        attach_crossview(pipe.dit)
        attach_larry(pipe.dit, WEIGHTS / "minimax_h3_turbo_v4_step600_ema.safetensors")
        place_model(pipe.dit, ["cuda:0", "cuda:1"], mlp_chunk_tokens=2048)
        pipe.device = torch.device("cuda:0")
        pipe.scheduler.set_timesteps(config["steps"], shift=config["video_shift"])
        pipe.scheduler_audio.set_timesteps(config["steps"], shift=config["audio_shift"])
        video, audio = noise["video_latents"].to("cuda:0"), noise["audio_latents"].to("cuda:0")
        context = move_context(context, "cuda:0")
        for i in range(config["steps"]):
            v, a = velocity(pipe, context, video, audio, float(pipe.scheduler.sigmas[i]),
                            float(pipe.scheduler_audio.sigmas[i]))
            video = pipe.scheduler.step(v, pipe.scheduler.timesteps[i], video)
            audio = pipe.scheduler_audio.step(a, pipe.scheduler_audio.timesteps[i], audio)
            print(f"Sampling {i+1}/{config['steps']}", flush=True)
        if not torch.isfinite(video).all():
            raise FloatingPointError("Nonfinite text-only video latent")
        video = video.cpu()
        del pipe, context, noise, audio, v, a
        gc.collect()
        torch.cuda.empty_cache()

        print("Decoding text-only result", flush=True)
        vae = load_pipe("Ref2VA", ("video_vae",), "cuda:0", MODEL)
        frames = vae.video_vae.decode_video(
            video.to("cuda:0"), dtype=torch.bfloat16, tiled=True, tile_size=256, tile_overlap=64
        ).cpu()
        with TemporaryDirectory(prefix="text-only-video-", dir=out) as temp:
            target = Path(temp) / "generated.mp4"
            write_video(frames, target, FRAMES, FPS)
            read_frames(target, FRAMES, FPS, WIDTH, HEIGHT)
            dump(out / "generation.json", config | {
                "generated_sha256": sha(target), "first_frame_sha256": None,
                "sampler": "native Ref2VA flow Euler, unanchored first frame",
            })
            target.replace(out / "generated.mp4")

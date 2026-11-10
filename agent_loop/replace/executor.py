"""Frame-zero vehicle replacement using tracked masks and an image-plane guide."""

import argparse
import gc
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .runtime import ROOT, dump, load, sha, sam_predictor

WIDTH, HEIGHT, FPS, FRAMES, SEED = 832, 480, 24, 124, 20261028


def read_frames(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        rate = cap.get(cv2.CAP_PROP_FPS)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape != (HEIGHT, WIDTH, 3):
                raise ValueError(f"Unexpected source shape: {frame.shape}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if len(frames) != FRAMES or abs(rate - FPS) > .001:
        raise ValueError(f"Expected {FRAMES} frames at {FPS} FPS: {path}")
    return np.stack(frames)


def projected_box(geometry, index, frame):
    center = geometry["centers"][index, frame]
    rotation = geometry["rotations"][index, frame]
    size = geometry["sizes"][index]
    signs = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])
    world = (signs * size) @ rotation.T + center
    camera = geometry["camera"][frame]  # OpenCV c2w, X right, Y down, Z forward.
    cam = (world - camera[:3, 3]) @ camera[:3, :3]
    if cam[:, 2].min() <= .5:
        return None
    uv = cv2.projectPoints(cam, np.zeros(3), np.zeros(3),
                           geometry["K"], geometry["distortion"])[0].reshape(-1, 2)
    return np.r_[uv.min(0), uv.max(0)]


def initialize(args, out):
    reference = getattr(args, "replacement_image", None)
    appearance = None
    if reference is not None:
        with Image.open(reference) as supplied:
            appearance = ImageOps.exif_transpose(supplied).convert("RGB")
    request = {
        "instruction": args.instruction, "source": str(args.source.resolve()),
        "source_sha256": sha(args.source.resolve()),
        "geometry": str(args.geometry.resolve()) if args.geometry else None,
        "geometry_sha256": sha(args.geometry.resolve()) if args.geometry else None,
        "tracking_cache": str(args.tracking_cache.resolve()) if args.tracking_cache else None,
        "tracking_cache_sha256": sha(args.tracking_cache.resolve()) if args.tracking_cache else None,
        "frames": FRAMES, "fps": FPS, "seed": SEED,
    }
    if getattr(args, "task_parameters", None) is not None:
        request["task_parameters"] = args.task_parameters
    if getattr(args, "conditioning", "appearance-guided") == "text-only":
        request["conditioning"] = "text-only"
    if reference is not None:
        request.update(
            conditioning="appearance-guided",
            replacement_image=str(reference.resolve()), replacement_image_sha256=sha(reference),
        )
    if (out / "request.json").exists():
        if load(out / "request.json") != request:
            raise ValueError("Resume inputs differ from the saved request")
        if appearance is not None:
            with Image.open(out / "replacement_reference.png") as saved:
                if not np.array_equal(np.array(saved.convert("RGB")), np.array(appearance)):
                    raise ValueError("Saved replacement reference image has changed")
        return
    if appearance is not None:
        appearance.save(out / "replacement_reference.png")
    frames = read_frames(args.source)
    Image.fromarray(frames[0]).save(out / "source_first_frame.png")
    if args.geometry:
        with np.load(args.geometry) as geometry:
            if geometry["camera"].shape != (FRAMES, 4, 4):
                raise ValueError("Geometry does not match the source frame count")
            candidates = []
            for index, object_id in enumerate(geometry["ids"]):
                if not geometry["valid"][index, 0]:
                    continue
                box = projected_box(geometry, index, 0)
                if box is None or box[2] <= 0 or box[0] >= WIDTH or box[3] <= 0 or box[1] >= HEIGHT:
                    continue
                if np.prod(box[2:] - box[:2]) < 25:
                    continue
                candidates.append({"object_id": str(object_id), "index": index,
                                   "bbox_xyxy": np.round(box, 2).tolist()})
    else:
        candidates = sam_candidates(out / "source_first_frame.png")
    if not candidates:
        raise ValueError("No visible candidate vehicles were detected in the first frame")
    dump(out / "candidates.json", candidates)
    dump(out / "request.json", request)


def sam_candidates(image):
    import torch

    model = sam_predictor()
    session = model.handle_request({"type": "start_session", "resource_path": str(image)})["session_id"]
    try:
        response = model.handle_request({
            "type": "add_prompt", "session_id": session, "frame_index": 0, "text": "vehicle"
        })["outputs"]
        ids = np.asarray(response["out_obj_ids"]).reshape(-1)
        masks = np.asarray(response["out_binary_masks"]).reshape(-1, HEIGHT, WIDTH).astype(bool)
        candidates = [
            {"object_id": f"sam-{int(object_id)}", "sam_instance_id": int(object_id),
             "bbox_xyxy": mask_box(mask).tolist(), "origin": "SAM3.1 vehicle detection"}
            for object_id, mask in zip(ids, masks) if mask.sum() >= 100
        ]
    finally:
        model.handle_request({"type": "close_session", "session_id": session})
        model.shutdown()
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return candidates


def mask_box(mask):
    # The visible target is one instance. Ignore disconnected segmentation speckles.
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        np.asarray(mask, dtype=np.uint8), connectivity=8
    )
    if count <= 1:
        raise ValueError("Empty instance mask")
    x, y, width, height, _ = stats[1:][np.argmax(stats[1:, cv2.CC_STAT_AREA])]
    return np.array([x, y, x + width, y + height], dtype=float)


def select_mask(outputs, bbox):
    masks = np.asarray(outputs["out_binary_masks"]).reshape(-1, HEIGHT, WIDTH).astype(bool)
    ids = np.asarray(outputs["out_obj_ids"]).reshape(-1)
    rectangle = np.zeros((HEIGHT, WIDTH), bool)
    x0, y0, x1, y1 = np.rint(bbox).astype(int)
    rectangle[max(0, y0):min(HEIGHT, y1), max(0, x0):min(WIDTH, x1)] = True
    scores = [float((mask & rectangle).sum() / max(1, (mask | rectangle).sum())) for mask in masks]
    if not scores or max(scores) < .15:
        raise ValueError("SAM did not ground the intended instance")
    index = int(np.argmax(scores))
    return masks[index], int(ids[index]), scores[index]


def segment_video(source, box, text="vehicle"):
    import torch

    model = sam_predictor()
    session = model.handle_request({"type": "start_session", "resource_path": str(source)})["session_id"]
    masks = np.zeros((FRAMES, HEIGHT, WIDTH), dtype=bool)
    seen = np.zeros(FRAMES, bool)
    try:
        first = model.handle_request({
            "type": "add_prompt", "session_id": session, "frame_index": 0, "text": text
        })
        initial, target, _ = select_mask(first["outputs"], box)
        masks[0], seen[0] = initial, True
        for entry in model.handle_stream_request({
            "type": "propagate_in_video", "session_id": session,
            "propagation_direction": "forward", "start_frame_idx": 0,
        }):
            i = int(entry["frame_index"])
            ids = np.asarray(entry["outputs"]["out_obj_ids"]).reshape(-1)
            match = np.flatnonzero(ids == target)
            if len(match):
                masks[i] = np.asarray(entry["outputs"]["out_binary_masks"])[match[0]].squeeze()
                seen[i] = True
    finally:
        model.handle_request({"type": "close_session", "session_id": session})
        model.shutdown()
        del model
        gc.collect()
        torch.cuda.empty_cache()
    if not seen.all():
        raise ValueError(f"SAM lost the target in frames {np.flatnonzero(~seen).tolist()}")
    return masks


def prepare(out):
    request, route = load(out / "request.json"), load(out / "route.json")
    if (out / "tracking.json").exists():
        saved = load(out / "tracking.json")
        if saved["mask_sha256"] != sha(out / "target_masks.npz"):
            raise ValueError("Saved target masks have changed")
        if saved["target_id"] != route["target_id"]:
            raise ValueError("Saved masks belong to a different routed target")
        return
    candidate = next(item for item in load(out / "candidates.json")
                     if item["object_id"] == route["target_id"])
    cache = Path(request["tracking_cache"]) if request["tracking_cache"] else None
    reused = False
    if cache is not None:
        sibling_source = cache.parent / "source.mp4"
        if not sibling_source.exists() or sha(sibling_source) != request["source_sha256"]:
            raise ValueError("Tracking cache is not bound to this source video")
        with np.load(cache) as old:
            reused = str(old["object_id"]) == route["target_id"]
            if reused:
                masks = old["masks"].astype(bool)
    if not reused:
        masks = segment_video(Path(request["source"]), candidate["bbox_xyxy"])
    if masks.shape != (FRAMES, HEIGHT, WIDTH) or masks.sum((1, 2)).min() < 100:
        raise ValueError("Invalid or incomplete target masks")
    inside = None
    if request["geometry"]:
        with np.load(request["geometry"]) as geometry:
            inside = []
            for i, mask in enumerate(masks):
                box = projected_box(geometry, candidate["index"], i)
                if box is None:
                    raise ValueError("Target has invalid source geometry")
                y, x = np.nonzero(mask)
                inside.append(float(((x >= box[0] - 6) & (x <= box[2] + 6)
                                     & (y >= box[1] - 6) & (y <= box[3] + 6)).mean()))
            if min(inside) < .75:
                raise ValueError("SAM masks do not agree with the selected source instance")
    boxes = np.stack([mask_box(mask) for mask in masks])
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    first_box = np.asarray(candidate["bbox_xyxy"], dtype=float)
    _, _, initial_overlap = select_mask(
        {"out_binary_masks": masks[:1], "out_obj_ids": np.array([0])}, first_box
    )
    np.savez_compressed(out / "target_masks.npz", masks=masks, object_id=route["target_id"])
    first = np.array(Image.open(out / "source_first_frame.png").convert("RGB"))
    first[masks[0]] = (first[masks[0]] * .5 + np.array([0, 255, 80]) * .5).astype(np.uint8)
    Image.fromarray(first).save(out / "target_selection.png")
    prompt = (
        route["first_frame_edit_prompt"] + "\n"
        f"Edit the supplied {WIDTH}x{HEIGHT} image. Replace only the selected vehicle at "
        f"pixel box {candidate['bbox_xyxy']}. Target description: {route['target_description']}. "
        f"Create one realistic {route['replacement_category']}. "
        "Keep its road contact close to the old vehicle's road contact and its travel direction. "
        "Retain the image framing and all other cars, "
        "road markings, background, sky, perspective, existing weather and lighting. No text or logos."
    )
    images = [{"role": "scene_edit_target", "path": str(out / "source_first_frame.png")}]
    if request.get("replacement_image"):
        if sha(Path(request["replacement_image"])) != request["replacement_image_sha256"]:
            raise ValueError("Customer replacement image has changed")
        prompt += (
            " Image 1 is the scene to edit. Image 2 is the supplied replacement vehicle reference. "
            "Match Image 2's vehicle body, color, cab/cargo layout and distinctive details, "
            "adapted to Image 1's camera view and lighting."
        )
        images.append({"role": "replacement_vehicle_reference",
                       "path": str(out / "replacement_reference.png")})
    dump(out / "image_edit_request.json", {
        "operation": "replace_selected_vehicle_in_first_frame",
        "input_image": str(out / "source_first_frame.png"), "mask_sequence": str(out / "target_masks.npz"),
        "target_id": route["target_id"], "target_bbox": candidate["bbox_xyxy"], "prompt": prompt,
        "input_images": images,
    })
    dump(out / "tracking.json", {
        "target_id": route["target_id"], "reused_sam_masks": reused,
        "mask_sha256": sha(out / "target_masks.npz"), "source_sha256": request["source_sha256"],
        "frames_present": FRAMES,
        "minimum_fraction_inside_projected_box": min(inside) if inside is not None else None,
        "grounding_source": "scene_geometry" if request["geometry"] else "SAM3.1 source-video detection",
        "bbox_method": "largest_connected_component",
        "initial_candidate_overlap": initial_overlap,
        "maximum_center_step_px": float(np.linalg.norm(np.diff(centers, axis=0), axis=1).max()),
    })


def segment_replacement(out, image):
    import torch

    route = load(out / "route.json")
    target = load(out / "image_edit_request.json")["target_bbox"]
    saved = out / "replacement_first_mask.npz"
    if saved.exists():
        with np.load(saved) as checkpoint:
            if str(checkpoint["image_sha256"]) != sha(image):
                raise ValueError("Replacement mask belongs to a different image edit")
            return checkpoint["mask"].astype(bool)
    model = sam_predictor()
    session = model.handle_request({"type": "start_session", "resource_path": str(image)})["session_id"]
    try:
        response = model.handle_request({
            "type": "add_prompt", "session_id": session, "frame_index": 0,
            "text": route["replacement_category"],
        })
        mask, _, score = select_mask(response["outputs"], target)
    finally:
        model.handle_request({"type": "close_session", "session_id": session})
        model.shutdown()
        del model
        gc.collect()
        torch.cuda.empty_cache()
    old = np.load(out / "target_masks.npz")["masks"][0]
    area_ratio = float(mask.sum() / old.sum())
    if not .65 < area_ratio < 6:
        raise ValueError(f"Replacement size is implausible for this pilot: area ratio {area_ratio:.2f}")
    dump(out / "replacement_segmentation.json", {
        "model": "SAM3.1", "text": route["replacement_category"],
        "source_box_iou": score, "area_ratio": area_ratio, "bbox_xyxy": mask_box(mask).tolist(),
    })
    np.savez_compressed(saved, mask=mask, image_sha256=sha(image))
    return mask


def write_frames(path, frames):
    import imageio.v2 as imageio

    with imageio.get_writer(str(path), fps=FPS, codec="libx264", quality=9,
                            macro_block_size=1, pixelformat="yuv420p") as writer:
        for frame in frames:
            writer.append_data(frame)


def compose_conditions(out):
    from scipy.signal import savgol_filter

    request = load(out / "request.json")
    source = read_frames(request["source"])
    edited = np.array(Image.open(out / "editor_output.png").convert("RGB"))
    mask = segment_replacement(out, out / "editor_output.png")
    source_masks = np.load(out / "target_masks.npz")["masks"]
    boxes = np.stack([mask_box(source_mask) for source_mask in source_masks])
    anchors = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]]
    scales = np.sqrt(
        (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        / ((boxes[0, 2] - boxes[0, 0]) * (boxes[0, 3] - boxes[0, 1]))
    )
    # Smooth segmentation pixel jitter. Preserve the source first-frame anchor.
    smooth = savgol_filter(np.c_[anchors, scales], 7, 2, axis=0)
    smooth[0] = np.r_[anchors[0], 1.]
    box = mask_box(mask)
    appearance_anchor = np.array([(box[0] + box[2]) / 2, box[3]])
    alpha = cv2.GaussianBlur(mask.astype(np.float32), (3, 3), .6)
    masks, edits, result = [], [], []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    for i, frame in enumerate(source):
        center, scale = smooth[i, :2], float(smooth[i, 2])
        matrix = np.array([[scale, 0, center[0] - scale * appearance_anchor[0]],
                           [0, scale, center[1] - scale * appearance_anchor[1]]], dtype=np.float32)
        warped = cv2.warpAffine(edited, matrix, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR)
        matte = cv2.warpAffine(alpha, matrix, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR)
        target = matte > .05
        hole = cv2.dilate(source_masks[i].astype(np.uint8), kernel)
        background = cv2.inpaint(frame, hole, 5, cv2.INPAINT_TELEA)
        rgb = np.rint(warped * matte[..., None] + background * (1 - matte[..., None])).astype(np.uint8)
        support = (hole > 0) | (matte > 0)
        if not np.array_equal(rgb[~support], frame[~support]):
            raise ValueError("Guide changed source pixels outside the intended edit region")
        result.append(rgb)
        masks.append(target)
        edits.append(support)
    result = np.stack(result)
    masks = np.stack(masks)
    np.savez_compressed(out / "edit_masks.npz", replacement=masks, affected=np.stack(edits))
    Image.fromarray(result[0]).save(out / "edited_first_frame.png")
    first_box = mask_box(masks[0]).astype(int)
    x0, y0 = np.maximum(first_box[:2] - 12, 0)
    x1, y1 = np.minimum(first_box[2:] + 12, [WIDTH, HEIGHT])
    if request.get("replacement_image"):
        shutil.copyfile(out / "replacement_reference.png", out / "appearance_reference.png")
    else:
        Image.fromarray(result[0, y0:y1, x0:x1]).save(out / "appearance_reference.png")
    write_frames(out / "edited_reference.mp4", result)
    read_frames(out / "edited_reference.mp4")
    dump(out / "conditions.json", {
        "route": "tracked image-plane appearance warp", "metric_3d_control": False,
        "route_sha256": sha(out / "route.json"),
        "source_sha256": request["source_sha256"],
        "editor_output_sha256": sha(out / "editor_output.png"),
        "edited_first_frame_sha256": sha(out / "edited_first_frame.png"),
        "edited_reference_sha256": sha(out / "edited_reference.mp4"),
        "edit_masks_sha256": sha(out / "edit_masks.npz"),
        "appearance_reference_sha256": sha(out / "appearance_reference.png"),
        "appearance_source": "user_reference" if request.get("replacement_image") else "edited_first_frame",
        "same_video_for_source_and_aligned_guide": True,
        "mask_binding": "RGB condition preparation",
        "maximum_anchor_smoothing_px": float(np.linalg.norm(smooth[:, :2] - anchors, axis=1).max()),
        "limitations": [
            "One first-frame appearance warp has limited viewpoint coverage.",
            "Local background holes use inpainting.",
            "Occlusions and generated trajectory require output review.",
        ],
    })


def h3_prompt(out):
    route = load(out / "route.json")
    category = route["replacement_category"]
    return (
        "crossview.\n"
        "subject_definitions:\n"
        f"<Subject 1> is the {category} appearance from <Picture 1>.\n"
        "<Video 1> is the edited driving reference with the selected vehicle already replaced.\n\n"
        "summary:\n"
        "[video editing + reference generation] The target video is an edited version of <Video 1>. "
        f"The selected vehicle is a {category} from the first frame, matching <Subject 1>.\n\n"
        "retention_analysis:\n"
        "<Subject 1> (appears in [Shot 1]): fully_preserved - retain its replacement appearance.\n"
        "<Video 1> (appears in [Shot 1]): fully_preserved - retain the road, traffic and camera motion.\n\n"
        "detailed_description:\n"
        "Photorealistic continuous dashcam video preserving the input weather, lighting and camera viewpoint.\n"
        f"[Shot 1] From the opening frame, the selected vehicle in <Video 1> is the {category} in <Subject 1>. "
        "It has the same visible body details, wheels, color and proportions "
        "as the appearance reference. The road and its initial composition follow <Video 1>. "
        f"{route['motion_description']} "
        "Keep the replacement a coherent vehicle throughout the clip, following its motion in the "
        "edited video guide. Maintain tire contact, plausible shadows and perspective as it travels "
        "along the road. Other traffic follows the reference. Preserve the uninterrupted forward "
        "dashcam viewpoint, without a cut or zoom.\n\n"
        "overall_soundscape:\nN/A\n\nnon_diegetic_music:\nN/A"
    )


def infer(out):
    import crossview_v2v as h3

    if (out / "generated.mp4").exists():
        checked = load(out / "generation.json")
        if sha(out / "generated.mp4") != checked["generated_sha256"]:
            raise ValueError("Existing generation hash mismatch")
        return
    (ROOT / "logs").mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="replacement-h3-", dir=ROOT / "logs") as temp:
        temp = Path(temp)
        for name in ("source.mp4", "warp.mp4"):
            (temp / name).symlink_to(out / "edited_reference.mp4")
        dump(temp / "config.json", {
            "task_type": "vehicle_replacement", "frames": FRAMES, "fps": FPS,
            "seed": SEED, "prompt": h3_prompt(out),
            "first_frame_anchor": str(out / "edited_first_frame.png"),
        })
        h3.OUT, h3.PROMPT, h3.SEED, h3.DEVICES = temp, h3_prompt(out), SEED, ["cuda:0", "cuda:1"]
        h3.infer(reference_image=out / "appearance_reference.png",
                 first_frame_image=out / "edited_first_frame.png")
        generation = load(temp / "generation.json")
        if generation["first_frame_sha256"] != sha(out / "edited_first_frame.png"):
            raise ValueError("H3 did not bind the intended first-frame image")
        generation["conditions_sha256"] = sha(out / "conditions.json")
        dump(out / "generation.json", generation)
        for name in ("guide_audit.json",):
            shutil.copyfile(temp / name, out / name)
        (temp / "generated.mp4").replace(out / "generated.mp4")


def generate(out, edited):
    if edited is not None:
        imported = Image.open(edited).convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
        imported_path = out / "editor_output.png"
        if imported_path.exists():
            if not np.array_equal(np.array(imported), np.array(Image.open(imported_path))):
                raise ValueError("Resume image differs from the previous editor output")
        else:
            imported.save(imported_path)
            dump(out / "image_editor.json", {
                "provider": "host image editor", "input_sha256": sha(out / "source_first_frame.png"),
                "returned_image_sha256": sha(edited), "normalized_image_sha256": sha(imported_path),
            })
    if not (out / "editor_output.png").exists():
        raise ValueError("Image editor output is missing")
    if not (out / "conditions.json").exists():
        dump(out / "state.json", {"stage": "preparing_conditions", "agent": "vehicle_replacement"})
        compose_conditions(out)
    conditions = load(out / "conditions.json")
    for key, path in (
        ("route_sha256", "route.json"), ("editor_output_sha256", "editor_output.png"),
        ("edited_first_frame_sha256", "edited_first_frame.png"),
        ("edited_reference_sha256", "edited_reference.mp4"),
        ("edit_masks_sha256", "edit_masks.npz"),
        ("appearance_reference_sha256", "appearance_reference.png"),
    ):
        if conditions[key] != sha(out / path):
            raise ValueError(f"Saved generation condition has changed: {path}")
    dump(out / "state.json", {"stage": "generating", "agent": "vehicle_replacement"})
    with (out / "h3.log").open("a") as log:
        subprocess.run(
            [sys.executable, "-m", "agent_loop.replace.executor", "infer", "--out", str(out)],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
        )



def text_only_prompt(out):
    route = load(out / "route.json")
    for key in ("target_description", "replacement_category", "replacement_description", "motion_description"):
        if not isinstance(route.get(key), str) or not route[key].strip():
            raise ValueError(f"Missing agent-authored {key}; plan this request in a new output directory")
    candidate = next((item for item in load(out / "candidates.json")
                      if item["object_id"] == route["target_id"]), None)
    if candidate is None:
        raise ValueError("Routed target is not a detected source candidate")
    category = route["replacement_category"]
    return (
        "crossview.\n"
        "subject_definitions:\n"
        "<Video 1> is the original driving video before the requested vehicle replacement.\n"
        f"<Subject 1> is the selected existing vehicle in <Video 1>: {route['target_description']}. "
        f"Its first-frame bounding box is {candidate['bbox_xyxy']} in pixel coordinates "
        f"(x_min, y_min, x_max, y_max) on the {WIDTH}x{HEIGHT} source frame. "
        "This box identifies the source instance at frame zero; follow that same instance's motion.\n\n"
        "summary:\n"
        "[video editing] The target video is an edited version of <Video 1>. "
        f"Replace <Subject 1> with a realistic {category} from the very first frame. "
        "The replacement appearance is specified entirely by text.\n\n"
        "retention_analysis:\n"
        "<Video 1>: partially_preserved - preserve its road, camera motion and other traffic.\n"
        "<Subject 1>: partially_preserved - preserve its lane and movement while replacing its body "
        f"with the requested {category}.\n\n"
        "detailed_description:\n"
        "Photorealistic continuous dashcam footage retaining the source weather and illumination.\n"
        f"[Shot 1] Identify <Subject 1> by its original appearance and first-frame location. "
        f"From the opening frame at 0.00 seconds, it is already a complete {category}. "
        f"Replacement appearance: {route['replacement_description']} "
        "The replacement remains the same coherent vehicle throughout the shot. "
        f"{route['motion_description']} "
        "Keep its tires in contact with the road and its scale and perspective plausible. "
        "Preserve the ego viewpoint, street layout, lane markings, lighting and all unrelated vehicles "
        "from <Video 1>. Keep the camera continuous, without cuts, zoom, logos or text overlays.\n\n"
        "overall_soundscape:\nN/A\n\nnon_diegetic_music:\nN/A"
    )


def generate_text_only(out):
    request = load(out / "request.json")
    if sha(Path(request["source"])) != request["source_sha256"]:
        raise ValueError("Source video has changed")
    manifest = {
        "conditioning": "text-only", "prompt_schema_version": 2, "source": request["source"],
        "route_sha256": sha(out / "route.json"),
        "source_sha256": request["source_sha256"], "prompt": text_only_prompt(out),
        "reference_video_count": 1, "reference_image_count": 0,
        "aligned_warp": None, "first_frame_anchor": None,
        "frames": FRAMES, "fps": FPS, "seed": SEED, "steps": 8,
        "video_shift": 12, "audio_shift": 3,
        "crossview_strength": .8, "acceleration": "Larry v4 step600 EMA",
    }
    config = out / "text_only_config.json"
    if config.exists() and load(config) != manifest:
        raise ValueError("Saved text-only generation inputs differ")
    dump(config, manifest)
    dump(out / "state.json", {"stage": "generating", "agent": "vehicle_replacement"})
    with (out / "h3.log").open("a") as log:
        subprocess.run(
            [sys.executable, "-m", "agent_loop.replace.executor", "infer-text-only", "--out", str(out)],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
        )


def infer_text_only(out):
    from ..ref2va import infer as infer_video_text

    infer_video_text(out, "text_only_config.json")


def compare(out, source, generated):
    import imageio.v2 as imageio

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    from PIL import ImageFont

    font = ImageFont.truetype(font_path, 24) if Path(font_path).exists() else ImageFont.load_default()
    banner = Image.new("RGB", (WIDTH * 2, 48), "#141923")
    draw = ImageDraw.Draw(banner)
    draw.text((16, 10), "Original", font=font, fill="white")
    draw.text((WIDTH + 16, 10), "Vehicle Replacement", font=font, fill="white")
    with imageio.get_writer(str(out / "comparison.mp4"), fps=FPS, codec="libx264", quality=8,
                            macro_block_size=1, pixelformat="yuv420p") as writer:
        for a, b in zip(source, generated):
            writer.append_data(np.concatenate([np.array(banner), np.concatenate([a, b], axis=1)]))


def verify(out):
    from .dispatch import QWEN_PYTHON

    dump(out / "state.json", {"stage": "verifying", "agent": "vehicle_replacement"})
    request = load(out / "request.json")
    source, generated = read_frames(request["source"]), read_frames(out / "generated.mp4")
    text_only = request.get("conditioning") == "text-only"
    if text_only:
        route = load(out / "route.json")
        candidate = next(item for item in load(out / "candidates.json")
                         if item["object_id"] == route["target_id"])
        x0, y0, x1, y1 = np.rint(candidate["bbox_xyxy"]).astype(int)
        roi = np.zeros((HEIGHT, WIDTH), np.uint8)
        roi[max(0, y0):min(HEIGHT, y1), max(0, x0):min(WIDTH, x1)] = 1
        target = np.broadcast_to(roi.astype(bool), (FRAMES, HEIGHT, WIDTH))
        expanded = cv2.dilate(roi, np.ones((181, 181), np.uint8)).astype(bool)
        affected = np.broadcast_to(expanded, target.shape)
    else:
        masks = np.load(out / "edit_masks.npz")
        affected, target = masks["affected"], masks["replacement"]
    error = np.abs(generated.astype(np.float32) - source.astype(np.float32)).mean(axis=3)
    outside_error = error[~affected]
    anchor_metrics = {
        "first_frame_sha256": None, "first_frame_anchor_mae": None, "first_frame_target_mae": None,
    }
    if not text_only:
        frame0 = np.array(Image.open(out / "edited_first_frame.png").convert("RGB"))
        anchor_error = np.abs(generated[0].astype(np.float32) - frame0.astype(np.float32)).mean(axis=2)
        anchor_metrics = {
            "first_frame_sha256": sha(out / "edited_first_frame.png"),
            "first_frame_anchor_mae": float(anchor_error.mean()),
            "first_frame_target_mae": float(anchor_error[target[0]].mean()),
        }
    metrics = {
        "frames": len(generated), "fps": FPS, "width": WIDTH, "height": HEIGHT,
        "generated_sha256": sha(out / "generated.mp4"),
        **anchor_metrics,
        "conditioning": "text-only" if text_only else "appearance-guided",
        "outside_edit_region_mae": float(outside_error.mean()) if outside_error.size else None,
        "note": "Pixel differences are diagnostics. Target identity and appearance require visual review.",
    }
    compare(out, source, generated)
    indices = [0, 24, 48, 72, 96, 123]
    # Show source and output crops around the full replacement footprint for visual review.
    sheet = Image.new("RGB", (320 * len(indices), 224 * 2), "#141923")
    draw = ImageDraw.Draw(sheet)
    for col, i in enumerate(indices):
        box = mask_box(target[i])
        margin = 90 if text_only else 45
        x0, y0 = np.maximum(np.floor(box[:2] - margin).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(box[2:] + margin).astype(int), [WIDTH, HEIGHT])
        for row, frames in enumerate((source, generated)):
            crop = Image.fromarray(frames[i, y0:y1, x0:x1])
            crop.thumbnail((320, 196))
            x, y = col * 320, row * 224
            sheet.paste(crop, (x, y + 28))
            draw.text((x + 4, y + 7), f"{'SOURCE' if row == 0 else 'OUTPUT'} frame {i}", fill="white")
    sheet.save(out / "review_frames.jpg", quality=92)
    Image.fromarray(generated[0]).save(out / "generated_first_frame.png")
    dump(out / "technical_checks.json", metrics)
    if not (out / "visual_review.json").exists():
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["HF_ENABLE_PARALLEL_LOADING"] = "true"
        env["HF_PARALLEL_LOADING_WORKERS"] = "6"
        with (out / "review.log").open("w") as log:
            subprocess.run(
                [QWEN_PYTHON, "-m", "agent_loop.replace", "review", "--out", str(out)],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True,
            )
    review = load(out / "visual_review.json")
    passed = (review["replacement_visible_in_first_frame"] and review["replacement_category_correct"]
              and review["target_identity_consistent"] and not review["wrong_vehicle_replaced"])
    if request.get("replacement_image"):
        passed = passed and review.get("reference_appearance_matched", False)
    dump(out / "state.json", {
        "stage": "complete" if passed else "completed_with_visual_issues",
        "agent": "vehicle_replacement", "technical_checks_passed": True,
        "visual_check_passed": passed, "video": str(out / "generated.mp4"),
    })
    print(f"Finished vehicle replacement, visual_check_passed={passed}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("infer", "infer-text-only"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "infer-text-only":
        infer_text_only(args.out.resolve())
    else:
        infer(args.out.resolve())


if __name__ == "__main__":
    main()

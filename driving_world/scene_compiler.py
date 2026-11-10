"""Compile a constrained semantic driving plan into proxy.py scene JSON.

Coordinates are world-frame meters, x forward, y left, z up; speeds are m/s.
The ego starts at the origin and drives straight. Events use absolute seconds.
"""

import math

import numpy as np

from .proxy import interpolate, validate as validate_scene

SIZES = {"vehicle": [4.5, 1.8, 1.6], "pedestrian": [0.6, 0.6, 1.7]}


def number(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def compile_plan(plan):
    duration = number(plan["duration"], "duration")
    speed = number(plan["ego_speed"], "ego_speed")
    if not 1 <= duration <= 20 or not 0 <= speed <= 30:
        raise ValueError("duration must be 1–20 seconds and ego_speed 0–30 m/s")
    events = plan["events"]
    if not isinstance(events, list) or not 1 <= len(events) <= 8:
        raise ValueError("events must contain 1–8 actions")
    scene = {
        "duration": duration,
        "fps": 24,
        "camera": {
            "width": 832,
            "height": 480,
            "fov_degrees": 85,
            "position": [1.5, 0, 1.5],
            "yaw": 0,
            "pitch": 0,
            "roll": 0,
        },
        "grid": {"extent": max(75, math.ceil(speed * duration + 40)), "spacing": 5},
        "ego": [
            {"t": 0, "position": [0, 0, 0], "yaw": 0},
            {"t": duration, "position": [speed * duration, 0, 0], "yaw": 0},
        ],
        "actors": [],
    }
    compiled_events = []
    for index, event in enumerate(events):
        kind = event["action"]
        if kind not in {"cut_in", "brake", "cross"}:
            raise ValueError(f"unsupported action: {kind}")
        start = number(event["start"], "start")
        end = number(event["end"], "end")
        distance = number(event["distance"], "distance")
        if not 0 <= start < end <= duration or not 3 <= distance <= 80:
            raise ValueError(
                f"event {index} ({kind}): start={start}, end={end}, duration={duration}, distance={distance}; require 0 <= start < end <= duration and distance 3–80 m"
            )
        actor_id = str(event.get("actor_id", f"actor_{index}"))
        # distance is measured in front of ego at the start of the action.
        x_start = speed * start + distance
        actor_speed = number(event.get("speed", speed), "speed")
        if not 0 <= actor_speed <= 35:
            raise ValueError("actor speed must be 0–35 m/s")
        actor_type = "pedestrian" if kind == "cross" else "vehicle"
        height = SIZES[actor_type][2] / 2
        if kind == "cross":
            direction = event.get("direction")
            if direction not in {"right_to_left", "left_to_right"}:
                raise ValueError("invalid cross direction")
            y0, y1 = (-5.0, 5.0) if direction == "right_to_left" else (5.0, -5.0)
            track = [
                {
                    "t": start,
                    "position": [x_start, y0, height],
                    "yaw": math.copysign(math.pi / 2, y1 - y0),
                },
                {
                    "t": end,
                    "position": [x_start, y1, height],
                    "yaw": math.copysign(math.pi / 2, y1 - y0),
                },
            ]
        else:
            x0 = x_start - actor_speed * start
            x_end = x_start + actor_speed * (end - start)
            if kind == "cut_in":
                direction = event.get("direction")
                if direction not in {"right_to_left", "left_to_right"}:
                    raise ValueError("invalid cut-in direction")
                y0 = -3.5 if direction == "right_to_left" else 3.5
                track = [
                    {"t": 0, "position": [x0, y0, height], "yaw": 0},
                    {"t": start, "position": [x_start, y0, height], "yaw": 0},
                    {"t": end, "position": [x_end, 0, height], "yaw": 0},
                    {
                        "t": duration,
                        "position": [x_end + actor_speed * (duration - end), 0, height],
                        "yaw": 0,
                    },
                ]
            else:
                final_speed = number(event.get("final_speed", 0), "final_speed")
                if not 0 <= final_speed <= actor_speed:
                    raise ValueError("brake final_speed must be between 0 and speed")
                # Sample constant-deceleration motion; linear interpolation preserves approximate motion.
                track = [
                    {"t": 0, "position": [x0, 0, height], "yaw": 0},
                    {"t": start, "position": [x_start, 0, height], "yaw": 0},
                ]
                for n in range(1, 9):
                    dt = (end - start) * n / 8
                    x = (
                        x_start
                        + actor_speed * dt
                        + 0.5 * (final_speed - actor_speed) / (end - start) * dt * dt
                    )
                    track.append(
                        {"t": start + dt, "position": [x, 0, height], "yaw": 0}
                    )
                x_end = track[-1]["position"][0]
                track.append(
                    {
                        "t": duration,
                        "position": [x_end + final_speed * (duration - end), 0, height],
                        "yaw": 0,
                    }
                )
        # Remove duplicated boundary keyframes at t=0 or t=duration.
        unique = {round(k["t"], 9): k for k in track}
        actor = {
            "id": actor_id,
            "type": actor_type,
            "size": SIZES[actor_type],
            "track": [unique[t] for t in sorted(unique)],
        }
        if kind == "cross":
            actor["start"] = start
        scene["actors"].append(actor)
        details = {
            "action": kind,
            "actor_id": actor_id,
            "start": start,
            "end": end,
            "target": {
                "cut_in": "enter ego lane",
                "brake": "reduce speed",
                "cross": "cross roadway",
            }[kind],
        }
        if kind == "cut_in":
            details.update(
                {
                    "start_lane": "right" if direction == "right_to_left" else "left",
                    "target_lane": "ego",
                    "duration_seconds": end - start,
                    "forward_speed_mps": actor_speed,
                }
            )
        elif kind == "cross":
            details.update(
                {"direction": direction, "walking_speed_mps": 10 / (end - start)}
            )
            if "speed" in event:
                details["ignored_input_speed_mps"] = event["speed"]
        else:
            details.update(
                {
                    "initial_speed_mps": actor_speed,
                    "final_speed_mps": final_speed,
                    "duration_seconds": end - start,
                    "deceleration_mps2": (actor_speed - final_speed) / (end - start),
                }
            )
        compiled_events.append(details)
    validate_scene(scene)
    return scene, compiled_events


def inspect_scene(scene, events):
    """Check generated trajectories; collision checks are approximate sampled AABBs."""
    warnings = []
    checks = []
    for event in events:
        actor = next(a for a in scene["actors"] if a["id"] == event["actor_id"])
        keyframes = actor["track"]
        velocities = [
            (np.array(b["position"]) - np.array(a["position"])) / (b["t"] - a["t"])
            for a, b in zip(keyframes, keyframes[1:])
        ]
        if velocities and max(np.linalg.norm(v) for v in velocities) > 35:
            warnings.append(f"{actor['id']}: speed exceeds 35 m/s")
        if any(abs(k["position"][1]) > 7 for k in keyframes):
            warnings.append(
                f"{actor['id']}: position is outside assumed road corridor ±7 m"
            )
        for i, (before, after) in enumerate(zip(velocities, velocities[1:])):
            dt = (keyframes[i + 2]["t"] - keyframes[i]["t"]) / 2
            if np.linalg.norm(after - before) / dt > 8:
                warnings.append(
                    f"{actor['id']}: estimated acceleration exceeds 8 m/s² near {keyframes[i+1]['t']:.1f}s"
                )
                break
        p0, _ = interpolate(actor["track"], event["start"])
        p1, _ = interpolate(actor["track"], event["end"])
        if event["action"] == "cut_in":
            ok = abs(p0[1]) > 1.75 and abs(p1[1]) < 0.3
        elif event["action"] == "cross":
            ok = p0[1] * p1[1] < 0
            crossing_speed = abs(p1[1] - p0[1]) / (event["end"] - event["start"])
            if crossing_speed > 3:
                warnings.append(
                    f"{actor['id']}: pedestrian speed {crossing_speed:.1f} m/s exceeds 3 m/s"
                )
        else:
            track = actor["track"]
            velocities = [
                (b["position"][0] - a["position"][0]) / (b["t"] - a["t"])
                for a, b in zip(track, track[1:])
            ]
            ok = min(velocities) < max(velocities) - 0.1
            deceleration = max(
                (
                    abs(b - a) / max(0.001, (track[i + 2]["t"] - track[i]["t"]) / 2)
                    for i, (a, b) in enumerate(zip(velocities, velocities[1:]))
                ),
                default=0,
            )
            if deceleration > 8:
                warnings.append(
                    f"{actor['id']}: estimated acceleration {deceleration:.1f} m/s² exceeds 8"
                )
        checks.append({**event, "trajectory_goal_met": bool(ok)})
    # Detect potential ego-actor overlap, without assuming all collisions are unintended.
    for t in np.arange(0, scene["duration"], 0.1):
        ego, _ = interpolate(scene["ego"], t)
        for actor in scene["actors"]:
            if t < actor.get("start", 0) or t >= actor.get("end", scene["duration"]):
                continue
            pos, _ = interpolate(actor["track"], t)
            if (
                abs(pos[0] - ego[0]) < (4.5 + actor["size"][0]) / 2
                and abs(pos[1] - ego[1]) < (1.8 + actor["size"][1]) / 2
            ):
                msg = f"possible ego/{actor['id']} overlap near {t:.1f}s (axis-aligned approximation)"
                if not any(actor["id"] in w and "overlap" in w for w in warnings):
                    warnings.append(msg)
    return {
        "checks": checks,
        "warnings": warnings,
        "compiler_assumptions": [
            "Straight road and constant-speed ego",
            "Lane centers y=0 and y=±3.5 m",
            "Crossing spans y=-5 to y=5 m",
            "Flat ground at z=0",
        ],
        "limitations": [
            "Approximate sampled checks; no map, actor-actor collision or continuous collision detection.",
            "Proxy semantics and H3 compliance require visual review.",
        ],
    }

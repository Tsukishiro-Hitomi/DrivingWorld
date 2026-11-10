"""Extract and check a deliberately limited set of explicit Chinese intent facts."""

import re

URGENT_MIN_DECELERATION = (
    4.0  # m/s²; a documented project default, not a legal standard.
)


def extract(request):
    """Extract only explicit supported phrases; unrecognized requirements remain unverified."""
    events = []
    if any(term in request for term in ("切入", "并入", "变道")):
        direction = None
        if re.search(r"右侧(?:车辆|车)|右边(?:车辆|车)|右车道", request):
            direction = "right_to_left"
        elif re.search(r"左侧(?:车辆|车)|左边(?:车辆|车)|左车道", request):
            direction = "left_to_right"
        events.append({"action": "cut_in", "direction": direction})
    if any(term in request for term in ("急刹", "急停", "刹车", "减速")):
        events.append(
            {
                "action": "brake",
                "urgent": any(term in request for term in ("急刹", "急停")),
                "stop": any(term in request for term in ("停车", "停下", "急停")),
            }
        )
    if any(term in request for term in ("横穿", "过马路", "穿过道路")):
        direction = None
        if re.search(r"(?:从|由)右(?:侧|边)?(?:向|往|至)左", request):
            direction = "right_to_left"
        elif re.search(r"(?:从|由)左(?:侧|边)?(?:向|往|至)右", request):
            direction = "left_to_right"
        events.append({"action": "cross", "direction": direction})
    # Only attribute a time when there is exactly one recognized event.
    match = re.search(r"(\d+(?:\.\d+)?)\s*秒(?:时|开始|后)", request)
    if match and len(events) == 1:
        events[0]["start"] = float(match.group(1))
    return {
        "events": events,
        "coverage": "partial",
        "unverified": [
            "Only supported explicit Chinese expressions are extracted; additional user requirements require human review."
        ],
        "defaults": {"urgent_min_deceleration_mps2": URGENT_MIN_DECELERATION},
    }


def compare(spec, plan):
    errors = []
    for expected in spec["events"]:
        matches = [
            item
            for item in plan.get("events", [])
            if item.get("action") == expected["action"]
        ]
        if not matches:
            errors.append(f"Missing requested action {expected['action']}")
            continue

        def complies(event):
            if (
                expected.get("direction")
                and event.get("direction") != expected["direction"]
            ):
                return False
            if (
                "start" in expected
                and abs(float(event["start"]) - expected["start"]) > 0.1
            ):
                return False
            if expected.get("stop") and float(event.get("final_speed", 0)) > 0.1:
                return False
            if expected.get("urgent"):
                decel = (
                    float(event.get("speed", plan["ego_speed"]))
                    - float(event.get("final_speed", 0))
                ) / (float(event["end"]) - float(event["start"]))
                if decel < URGENT_MIN_DECELERATION:
                    return False
            return True

        if not any(complies(event) for event in matches):
            detail = f"Action {expected['action']} violates explicit intent {expected}."
            if expected.get("urgent"):
                event = matches[0]
                initial = float(event.get("speed", plan["ego_speed"]))
                final = float(event.get("final_speed", 0))
                latest = (
                    float(event["start"]) + (initial - final) / URGENT_MIN_DECELERATION
                )
                detail += (
                    f" With speed={initial}, final_speed={final}, start={event['start']}, "
                    f"end must be <= {latest:.3f} for >= {URGENT_MIN_DECELERATION} m/s² deceleration."
                )
            errors.append(detail)
    return errors

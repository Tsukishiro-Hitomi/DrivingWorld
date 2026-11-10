"""Apply evidence limits without changing the model's recorded judgments."""

from copy import deepcopy
import re


def aggregate(statuses):
    values = list(statuses)
    if not values or any(v not in {"PASS", "FAIL", "UNKNOWN"} for v in values):
        raise ValueError("Invalid acceptance statuses")
    return "FAIL" if "FAIL" in values else ("UNKNOWN" if "UNKNOWN" in values else "PASS")


def assess(result, criterion_ids, observed_frames=None, frame_count=None, full_sequence_ids=()):
    # Canonicalize equivalent representations only; never change a model verdict/reason.
    result = deepcopy(result)
    normalization = []
    if isinstance(result.get("limitations"), dict):
        result["raw_limitations"] = deepcopy(result["limitations"])
        result["limitations"] = [result["limitations"]]
        normalization.append("wrapped a single structured limitation in an array")
    if observed_frames is not None and isinstance(result.get("checks"), dict):
        original_checks = deepcopy(result["checks"])
        for check in result["checks"].values():
            if not isinstance(check, dict) or not isinstance(check.get("evidence_frames"), list):
                continue
            converted = []
            for value in check["evidence_frames"]:
                if isinstance(value, str):
                    label = re.fullmatch(r"OUTPUT(?: frame)? ([0-9]+) / ([0-9]+(?:\.[0-9]+)?)s", value.strip())
                    if label:
                        value = int(label.group(1))
                    elif re.fullmatch(r"[0-9]+", value.strip()):
                        value = int(value.strip())
                converted.append(value)
            check["evidence_frames"] = converted
        if result["checks"] != original_checks:
            result["raw_checks"] = original_checks
            normalization.append("converted explicit output-frame labels to integer IDs")
    if normalization:
        result["format_normalization"] = normalization
    ids = set(criterion_ids)
    checks = result.get("checks")
    if not isinstance(checks, dict) or set(checks) != ids:
        raise ValueError("Review must cover exactly the required criteria")
    for key, check in checks.items():
        if (not isinstance(check, dict) or check.get("status") not in {"PASS", "FAIL", "UNKNOWN"}
                or not isinstance(check.get("reason"), str) or not check["reason"].strip()):
            raise ValueError("Invalid review check: " + key)
        if observed_frames is not None:
            evidence = check.get("evidence_frames")
            if (not isinstance(evidence, list) or any(type(i) is not int or i not in observed_frames for i in evidence)
                    or (check["status"] in {"PASS", "FAIL"} and not evidence)):
                raise ValueError("Check cites missing or unobserved frame evidence: " + key)
    limits = result.get("limitations")
    if not isinstance(limits, list):
        raise ValueError("Structured limitations are required")
    effective = deepcopy(checks)
    for limit in limits:
        if not isinstance(limit, dict) or set(limit) != {"criteria", "impact", "reason"}:
            raise ValueError("Each limitation must map its criteria, impact and reason")
        affected = limit["criteria"]
        if (not isinstance(affected, list) or any(not isinstance(i, str) or i not in ids for i in affected)
                or limit["impact"] not in {"informational", "unknown", "fail"}
                or not isinstance(limit["reason"], str) or not limit["reason"].strip()
                or (limit["impact"] != "informational" and not affected)):
            raise ValueError("Invalid limitation mapping")
        for key in affected:
            value = effective[key]
            if limit["impact"] == "fail":
                value["status"] = "FAIL"
            elif limit["impact"] == "unknown" and value["status"] != "FAIL":
                value["status"] = "UNKNOWN"
            value["reason"] += " Limitation: " + limit["reason"]
    if not set(full_sequence_ids) <= ids:
        raise ValueError("Unknown full-sequence criterion")
    if full_sequence_ids and (observed_frames is None or frame_count is None):
        raise ValueError("Full-sequence acceptance requires evidence coverage metadata")
    for key in full_sequence_ids:
        if set(observed_frames) != set(range(frame_count)) and effective[key]["status"] == "PASS":
            effective[key]["status"] = "UNKNOWN"
            effective[key]["reason"] += " Full-sequence continuity is not established by sampled images."
    return dict(result, effective_checks=effective,
                sampled_verdict=aggregate(v["status"] for v in checks.values()),
                acceptance_verdict=aggregate(v["status"] for v in effective.values()))

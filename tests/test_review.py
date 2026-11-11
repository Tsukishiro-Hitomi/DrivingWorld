import unittest

from driving_world.contracts.review import (
    OBJECT_CHECK_NAMES,
    validate_event_verdict,
    validate_object_verdict,
)


def object_verdict():
    return {
        "event_id": "cut_in_01",
        "status": "observed",
        "confidence": 0.9,
        "observed_interval_seconds": [1.1, 2.9],
        "checks": {
            "subject_visible": True,
            "motion_observed": True,
            "direction_correct": True,
            "timing_valid": True,
            "event_completed": True,
            "matches_proxy_trajectory": True,
        },
        "evidence_frame_indices": [1, 4, 8],
        "rationale": "The rendered vehicle follows the proxy lane change.",
    }


class ReviewContractTests(unittest.TestCase):
    def test_object_verdict_requires_six_checks(self):
        payload = object_verdict()
        del payload["checks"]["matches_proxy_trajectory"]
        with self.assertRaises(ValueError):
            validate_object_verdict(payload, 4)

    def test_object_verdict_accepts_six_checks(self):
        verdict = validate_object_verdict(object_verdict(), 4)
        self.assertEqual(verdict["status"], "observed")
        self.assertEqual(set(verdict["checks"]), OBJECT_CHECK_NAMES)

    def test_event_verdict_keeps_five_checks(self):
        payload = object_verdict()
        del payload["checks"]["matches_proxy_trajectory"]
        verdict = validate_event_verdict(payload, 4)
        self.assertEqual(verdict["status"], "observed")

    def test_indeterminate_allows_null_checks(self):
        payload = object_verdict()
        payload["status"] = "indeterminate"
        payload["confidence"] = 0.3
        payload["observed_interval_seconds"] = None
        payload["checks"] = {name: None for name in OBJECT_CHECK_NAMES}
        payload["evidence_frame_indices"] = []
        payload["rationale"] = "Cannot determine from the supplied frames."
        verdict = validate_object_verdict(payload, 4)
        self.assertEqual(verdict["status"], "indeterminate")


if __name__ == "__main__":
    unittest.main()

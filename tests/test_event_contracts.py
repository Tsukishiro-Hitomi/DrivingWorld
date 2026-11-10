import unittest

from driving_world.contracts.events import validate_event_spec, validate_event_verdict


EVENT = {
    "event_id": "cut_in_01",
    "kind": "vehicle_lane_change",
    "subject_type": "vehicle",
    "time_window_seconds": [1.0, 3.0],
    "direction": "right_to_left",
    "target": "ego_lane",
}


class EventContractTests(unittest.TestCase):
    def test_valid_event_and_verdict(self):
        event = validate_event_spec(EVENT)
        self.assertEqual(event["kind"], "vehicle_lane_change")
        verdict = validate_event_verdict(
            {
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
                },
                "evidence_frame_indices": [1, 4, 8],
                "rationale": "The vehicle moves from the right into the ego lane.",
            },
            4,
        )
        self.assertEqual(verdict["status"], "observed")

    def test_observed_requires_all_checks(self):
        bad = {
            "event_id": "cut_in_01",
            "status": "observed",
            "confidence": 0.9,
            "observed_interval_seconds": [1.1, 2.9],
            "checks": {name: True for name in ("subject_visible", "motion_observed", "direction_correct", "timing_valid", "event_completed")},
            "evidence_frame_indices": [],
            "rationale": "Evidence.",
        }
        bad["checks"]["direction_correct"] = False
        with self.assertRaises(ValueError):
            validate_event_verdict(bad, 4)

    def test_subject_kind_mismatch_is_rejected(self):
        bad = dict(EVENT, subject_type="pedestrian")
        with self.assertRaises(ValueError):
            validate_event_spec(bad)


if __name__ == "__main__":
    unittest.main()

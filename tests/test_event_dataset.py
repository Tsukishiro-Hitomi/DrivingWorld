import unittest
from pathlib import Path

from driving_world.datasets.event_judge import load_manifest, validate_record


def record(record_id="sample_01"):
    return {
        "id": record_id,
        "split": "test",
        "video": "videos/sample_01.mp4",
        "event": {
            "event_id": "cross_01",
            "kind": "pedestrian_crossing",
            "subject_type": "pedestrian",
            "time_window_seconds": [1, 4],
            "direction": "right_to_left",
        },
        "annotation": {
            "event_id": "cross_01",
            "status": "indeterminate",
            "confidence": 0.4,
            "observed_interval_seconds": None,
            "checks": {
                "subject_visible": None,
                "motion_observed": None,
                "direction_correct": None,
                "timing_valid": None,
                "event_completed": None,
            },
            "evidence_frame_indices": [1, 4],
            "rationale": "The pedestrian is occluded during the requested interval.",
        },
    }


class EventDatasetTests(unittest.TestCase):
    def test_record_is_validated(self):
        self.assertEqual(validate_record(record())["id"], "sample_01")

    def test_duplicate_ids_are_rejected(self):
        manifest = Path(__file__).parent / "fixtures" / "duplicate_event_manifest.jsonl"
        with self.assertRaises(ValueError):
            load_manifest(manifest)


if __name__ == "__main__":
    unittest.main()

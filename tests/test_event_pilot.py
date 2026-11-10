import unittest

from driving_world.datasets.pilot import build_candidates


class PilotDatasetTests(unittest.TestCase):
    def test_balanced_candidate_matrix(self):
        candidates = build_candidates(2)
        self.assertEqual(len(candidates), 24)
        self.assertEqual({item.proxy_relation for item in candidates}, {"observed", "no_motion", "wrong_direction", "wrong_timing"})
        self.assertEqual({item.event["kind"] for item in candidates}, {"vehicle_lane_change", "pedestrian_crossing", "vehicle_hard_brake"})

if __name__ == "__main__":
    unittest.main()

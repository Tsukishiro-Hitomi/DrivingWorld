import unittest

from agent_loop.weather.rain import audit_target, normalize_review
from agent_loop.weather.sensor import apply_plan_patch, review_checks
from agent_loop.weather.profiles import h3_prompt, validate_plan, planning_profiles
from agent_loop.weather.tests.test_sensor import camera_plan


class RainIntensityTests(unittest.TestCase):
    def plan(self):
        return audit_target(camera_plan())

    def review(self, status):
        return dict(checks={key: dict(status="PASS", reason="Visible") for key in review_checks(self.plan())},
                    rain_evidence={key: dict(status=status, reason="Observable evidence")
                                   for key in ["dense_airborne_rain", "impact_or_spray", "depth_rain_veil"]})

    def test_wet_road_and_lens_drops_cannot_pass_airborne_intensity(self):
        review = self.review("PASS")
        review["rain_evidence"]["dense_airborne_rain"] = dict(status="FAIL", reason="Only wet asphalt and glass drops are visible")
        normalize_review(review)
        self.assertEqual(review["checks"]["precipitation_intensity"]["status"], "FAIL")
        self.assertEqual(review["checks"]["weather_match"]["status"], "FAIL")
        self.assertEqual(review["checks"]["sensor_effect_presence"]["status"], "PASS")

    def test_unknown_evidence_is_not_converted_to_success(self):
        result = normalize_review(self.review("UNKNOWN"))
        self.assertEqual(result["checks"]["weather_match"]["status"], "UNKNOWN")

    def test_all_independent_rain_cues_are_required(self):
        self.assertEqual(normalize_review(self.review("PASS"))["checks"]["precipitation_intensity"]["status"], "PASS")
        review = self.review("PASS")
        review["rain_evidence"].pop("depth_rain_veil")
        with self.assertRaises(ValueError):
            normalize_review(review)

    def test_v5_compiles_airborne_rain_and_active_camera_water_separately(self):
        plan = self.plan()
        validate_plan(plan, 124/24)
        text = h3_prompt(plan)
        self.assertIn("torrential downpour", text)
        self.assertIn("moving sheets", text)
        self.assertIn("continuously strike", text)
        self.assertIn("Fresh impacts continuously replenish", text)
        self.assertIn("Original source observations: RGB driving reference with observed day and dry", text)
        self.assertEqual(len(review_checks(plan)), 13)

    def test_rain_and_runoff_plan_patches_reach_the_prompt(self):
        plan = self.plan()
        patched = apply_plan_patch(plan, {"rain_controls": {"surface_impact": "very_strong", "vehicle_spray": "very_strong"},
                                          "sensor_effects": {"water_activity": "very_high"}}, 124/24)
        self.assertNotEqual(h3_prompt(plan), h3_prompt(patched))
        self.assertEqual(patched["edits"], plan["edits"])
        self.assertIn("forceful impacts", h3_prompt(patched))

    def test_repair_cannot_weaken_torrential_target(self):
        with self.assertRaises(ValueError):
            apply_plan_patch(self.plan(), {"rain_controls": {"airborne_density": "light"}}, 124/24)
        plan = self.plan()
        plan["rain_controls"]["airborne_density"] = "light"
        with self.assertRaises(ValueError):
            validate_plan(plan, 124/24)

    def test_current_policies_do_not_mutate_legacy_visibility(self):
        self.assertEqual(planning_profiles(5)["heavy_rain"]["environment"]["visibility"], "poor")
        self.assertEqual(planning_profiles(4)["heavy_rain"]["environment"]["visibility"], "reduced")


if __name__ == "__main__":
    unittest.main()

import unittest

from driving_world.scene_compiler import compile_plan, inspect_scene
from driving_world.plan_scene import semantic_warnings
from driving_world.intent import compare, extract


class CompilerTests(unittest.TestCase):
    def test_three_actions(self):
        for action in ("cut_in", "brake", "cross"):
            with self.subTest(action=action):
                plan = {"duration": 8, "ego_speed": 4, "events": [
                    {"actor_id": "target", "action": action, "start": 1, "end": 6,
                     "distance": 20, "speed": 4, "final_speed": 1,
                     "direction": "right_to_left"}]}
                scene, events = compile_plan(plan)
                self.assertTrue(inspect_scene(scene, events)["checks"][0]["trajectory_goal_met"])

    def test_invalid_time_and_action(self):
        for event in ({"action": "cross", "start": 5, "end": 2, "distance": 10},
                      {"action": "fly", "start": 1, "end": 2, "distance": 10}):
            with self.assertRaises(ValueError):
                compile_plan({"duration": 6, "ego_speed": 4, "events": [event]})

    def test_duplicate_actor_ids_rejected(self):
        event = {"actor_id": "car", "action": "brake", "start": 1, "end": 3, "distance": 15}
        with self.assertRaises(ValueError):
            compile_plan({"duration": 6, "ego_speed": 4, "events": [event, event]})

    def test_right_side_reversal_is_flagged(self):
        warnings = semantic_warnings("右侧车辆切入", {"events": [
            {"action": "cut_in", "direction": "left_to_right"}]})
        self.assertTrue(any("right-side" in warning for warning in warnings))

    def test_right_side_reversal_blocks(self):
        spec = extract("右侧车辆在2秒时向左切入本车道")
        plan = {"ego_speed": 5, "events": [{"action": "cut_in", "direction": "left_to_right",
                "start": 2, "end": 5}]}
        self.assertTrue(compare(spec, plan))
        plan["events"][0]["direction"] = "right_to_left"
        self.assertFalse(compare(spec, plan))

    def test_urgent_braking_threshold(self):
        spec = extract("前车在2秒开始急刹减速至停车")
        event = {"action": "brake", "start": 2, "end": 10, "speed": 10, "final_speed": 0}
        plan = {"ego_speed": 10, "events": [event]}
        self.assertTrue(compare(spec, plan))
        event["end"] = 4
        self.assertFalse(compare(spec, plan))


if __name__ == "__main__":
    unittest.main()

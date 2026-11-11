import unittest

from PIL import Image

from driving_world.judge.sampling import _positions, side_by_side


class SamplingHelperTests(unittest.TestCase):
    def test_positions_covers_endpoints(self):
        self.assertEqual(_positions(0, 10, 3), {0, 5, 10})

    def test_positions_single(self):
        self.assertEqual(_positions(4, 4, 8), {4})

    def test_side_by_side_width_and_divider(self):
        left = Image.new("RGB", (10, 20), (255, 0, 0))
        right = Image.new("RGB", (30, 20), (0, 0, 255))
        combined = side_by_side(left, right, divider_width=4)
        self.assertEqual(combined.size, (44, 20))
        self.assertEqual(combined.getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(combined.getpixel((12, 0)), (255, 255, 255))
        self.assertEqual(combined.getpixel((14, 0)), (0, 0, 255))


if __name__ == "__main__":
    unittest.main()

import unittest

from src.data_gen.color import Color


class TestColorLiterals(unittest.TestCase):
    def setUp(self) -> None:
        self.black = Color.BLACK

    def test_black_is_tuple(self):
        self.assertIsInstance(self.black, tuple)

    def test_black_is_0_0_0(self):
        rgb_black = (0, 0, 0)
        self.assertTupleEqual(self.black, rgb_black)

    def test_solid_is_255(self):
        self.assertEqual(Color.SOLID, 255)

    def test_transparent_is_0(self):
        self.assertEqual(Color.TRANSPARENT, 0)

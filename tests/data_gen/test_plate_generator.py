import unittest

from src.data_gen.label_properties import PlateLabel
from src.data_gen.plate_generator import PlateGenerator
from src.data_gen.plate_properties import RectangularPlate


class TestGenerateLabel(unittest.TestCase):
    def setUp(self) -> None:
        self.plate_gen = PlateGenerator(RectangularPlate, PlateLabel)
        self.valid_pos = PlateLabel.x_pos[0]

    def test_lowercase_letter_raises_value_error(self):
        lowercase_letter = 'c'
        with self.assertRaises(ValueError):
            self.plate_gen.generate_label(lowercase_letter, self.valid_pos)

    def test_int_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.plate_gen.generate_label(1, self.valid_pos)

    def test_invalid_x_position_raises_value_error(self):
        invalid_x_position = 6
        with self.assertRaises(ValueError):
            self.plate_gen.generate_label('A', invalid_x_position)

    def test_returns_string(self):
        result = self.plate_gen.generate_label('A', self.valid_pos)
        self.assertIsInstance(result, str)

    def test_character_in_middle_with_regex(self):
        expression = '^[A-Z0-9]{3}C[A-Z0-9]{3}$'  # looks for '___C___'
        position = PlateLabel.x_pos[3]
        character = 'C'

        result = self.plate_gen.generate_label(character, position)

        self.assertRegex(result, expression, 'Should be ___C___')

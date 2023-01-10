import unittest

from data_gen import PlateGenerator, RectangularPlate


class TestPlateGeneratorMethods(unittest.TestCase):
    def test_generate_label(self):
        # Invalid Character
        with self.assertRaises(ValueError):
            PlateGenerator(RectangularPlate).generate_label(
                'c',  # invalid
                RectangularPlate.x_pos[0]
            )

        # Invalid Position
        with self.assertRaises(ValueError):
            PlateGenerator(RectangularPlate).generate_label(
                'A',
                6  # invalid
            )

        label = PlateGenerator(RectangularPlate).generate_label(
            'A',
            RectangularPlate.x_pos[0]
        )
        # Correct Position
        self.assertTrue(label[0] == 'A')
        # Incorrect Position
        self.assertFalse(label[0] != 'A')

    def test_single_plate(self):
        # Invalid Character
        with self.assertRaises(ValueError):
            PlateGenerator(RectangularPlate).generate_plate(
                'c',  # invalid
                RectangularPlate.x_pos[0]
            )

        # Invalid Position
        with self.assertRaises(ValueError):
            PlateGenerator(RectangularPlate).generate_plate(
                'A',
                6  # invalid
            )


if __name__ == '__main__':
    unittest.main()

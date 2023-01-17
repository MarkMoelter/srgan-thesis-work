import unittest

import src.accuracy.accuracy as acc


class TestPredictionAccuracy(unittest.TestCase):
    def test_returns_float(self):
        prediction = ''
        label = ''

        result = acc.calc_prediction_accuracy(label, prediction)

        self.assertIsInstance(result, float)

    def test_nonstring_prediction_raises_type_error(self):
        prediction = 1
        label = ''

        with self.assertRaises(TypeError):
            acc.calc_prediction_accuracy(label, prediction)

    def test_nonstring_label_raises_type_error(self):
        prediction = ''
        label = 1

        with self.assertRaises(TypeError):
            acc.calc_prediction_accuracy(label, prediction)

    def test_same_label_and_prediction_returns_one(self):
        same_string = 'HELLO WORLD'

        result = acc.calc_prediction_accuracy(same_string, same_string)

        self.assertEqual(result, 1)

    def test_one_character_difference_with_strings_of_length_two(self):
        label = 'AB'
        prediction = 'AC'

        result = acc.calc_prediction_accuracy(label, prediction)

        self.assertAlmostEqual(result, 0.5)

    def test_prediction_trailing_whitespace_returns_same_as_no_whitespace(self):
        whitespace_prediction = 'ABCD '
        regular_prediction = 'ABCD'
        label = 'ABCE'

        whitespace_result = acc.calc_prediction_accuracy(label, whitespace_prediction)
        regular_result = acc.calc_prediction_accuracy(label, regular_prediction)

        self.assertEqual(whitespace_result, regular_result)

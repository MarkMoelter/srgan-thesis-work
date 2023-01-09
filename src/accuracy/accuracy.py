import logging
import statistics
from itertools import zip_longest

import pandas as pd

logger = logging.getLogger(__name__)


def prediction_accuracy(label: str, prediction: str) -> float:
    """
    Compute the accuracy between a label and a prediction.

    :param label: The correct label.
    :param prediction: The prediction to be checked for accuracy.
    :return: The accuracy between the two strings.
    """

    # check if not string
    if type(prediction) is not str:
        error_mes = f'{prediction} is not a string'
        logger.error(error_mes)
        raise ValueError(error_mes)

    # check if identical
    if label == prediction:
        return 1

    # element to element comparison
    num_matches = 0
    for c1, c2 in zip_longest(label, prediction):

        # add one if identical
        if c1 == c2:
            num_matches += 1

    return num_matches / len(label)


class Accuracy:
    def __init__(self, df: pd.DataFrame, label_col_name):
        self.df = df
        self.label_col_val = self.df[label_col_name].tolist()

    @staticmethod
    def string_accuracy(label: str, prediction: str) -> float:
        """
        Compute the accuracy between two strings.
        :param label: The correct label.
        :param prediction: The prediction to be checked for accuracy.
        :return: The accuracy between the two strings.
        """

        # check if not string
        if type(prediction) is not str:
            error_mes = f'{prediction} is not a string'
            logger.error(error_mes)
            raise ValueError(error_mes)

        # check if identical
        if label == prediction:
            return 1

        # element to element comparison
        num_matches = 0
        for c1, c2 in zip_longest(label, prediction):

            # add one if identical
            if c1 == c2:
                num_matches += 1

        return num_matches / len(label)

    def column_accuracy(self, pred_list: list) -> float:
        """
        Calculate the accuracy between 2 lists of equal length.
        :param pred_list: List of predictions
        :return: Average accuracy of the lists
        """

        # check if not list
        both_lists = isinstance(self.label_col_val, list) and isinstance(pred_list, list)
        if not both_lists:
            logger.error('Only lists are allowed')
            raise TypeError('Only lists are allowed')

        # if lengths are equal
        len_is_equal = len(self.label_col_val) == len(pred_list)
        if not len_is_equal:
            logger.error('Lists are not the same length')
            raise ValueError('Lists are not the same length')

            # accuracy between strings as lists
        acc_list = [
            self.string_accuracy(label, predict)
            for label, predict in zip(self.label_col_val, pred_list)
        ]

        # return the mean of all string's accuracy
        return statistics.mean(acc_list)

    def accuracy(self, pred_col_names: list[str]) -> list[float]:
        """
        Collect the list of all accuracies of each of the column names.

        :param pred_col_names: List of column names to compare with the labels
        :return: List of the specified column's accuracy
        """

        return [
            self.column_accuracy(pred_list=self.df[col_name].tolist())
            for col_name in pred_col_names
        ]

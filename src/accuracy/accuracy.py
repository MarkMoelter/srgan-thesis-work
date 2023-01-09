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


def column_accuracy(labels: list[str], predictions: list[str]) -> float:
    """
    Calculate the accuracy between 2 lists of equal length.

    :param labels: List of labels.
    :param predictions: List of predictions.
    :return: Average accuracy between the labels and the predictions.
    """

    # check if not list
    both_lists = isinstance(labels, list) and isinstance(predictions, list)
    if not both_lists:
        logger.error('Only lists are allowed')
        raise TypeError('Only lists are allowed')

    # if lengths are equal
    len_is_equal = len(labels) == len(predictions)
    if not len_is_equal:
        logger.error('Lists are not the same length')
        raise ValueError('Lists are not the same length')

        # accuracy between strings as lists
    acc_list = [
        prediction_accuracy(label, predict)
        for label, predict in zip(labels, predictions)
    ]

    # return the mean of all string's accuracy
    return statistics.mean(acc_list)


def accuracy(
        dataframe: pd.DataFrame,
        labels: list[str],
        prediction_headers: list[str]
) -> list[float]:
    """
    Collect the list of all accuracies of each of the column names.

    :param dataframe: The dataframe with the plate labels and predictions.
    :param labels: The list of label values for each plate.
    :param prediction_headers: List of column names for different predictions
     to compare with the labels.
    :return: List of the specified column's accuracy.
    """

    return [
        column_accuracy(labels, predictions=dataframe[col_name].tolist())
        for col_name in prediction_headers
    ]

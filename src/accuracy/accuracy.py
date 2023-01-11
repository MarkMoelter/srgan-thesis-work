import logging
import statistics
from itertools import zip_longest

import pandas as pd

import constants as c

logger = logging.getLogger(__name__)


def calculate_prediction_accuracy(label: str, prediction: str) -> float:
    """
    Compute the accuracy between a label and a prediction.

    :param label: The correct label.
    :param prediction: The prediction to be checked for accuracy.
    :return: The accuracy between the two strings.
    """

    # check if not string
    if not isinstance(prediction, str):
        raise ValueError(f'{prediction} is not a string')

    # check if identical
    if label == prediction:
        return 1

    # element to element comparison
    num_matches = 0
    for label_char, prediction_char in zip_longest(label, prediction):
        if label_char != prediction_char:
            continue

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
        calculate_prediction_accuracy(label, predict)
        for label, predict in zip(labels, predictions)
    ]

    # return the mean of all string's accuracy
    return statistics.mean(acc_list)


def accuracy(
        plate_dataframe: pd.DataFrame,
        headers: list[str],
        label_header: str = 'Real Plates'
) -> dict[str, float]:
    """
    Calculate the accuracy of OCR predictions given the dataframe of
     labels and predictions, and the column headers to be calculated.

    :param plate_dataframe: The dataframe with the plate labels and predictions.
    :param headers: List of column headers to process.
    :param label_header: The column header for the plate labels.
    :return: A dictionary containing the accuracy values.
    """

    # collect the plate labels
    labels = plate_dataframe[label_header].tolist()

    # iterate over each column
    accuracy_dict = {}
    for header in headers:
        # collect the plate predictions
        predictions: list[str] = plate_dataframe[header].tolist()
        accuracy_value: float = column_accuracy(labels, predictions)

        # append results to dictionary
        accuracy_dict[header] = accuracy_value
        logger.debug(f'{header} Accuracy: {accuracy_value}')

    return accuracy_dict

import logging
import statistics
from itertools import zip_longest

logger = logging.getLogger(__name__)

class Accuracy:
    def __init__(self, DataFrame, label_col_name):
        # Dataframe
        self.df = DataFrame

        # label attributes
        self.label_col_name = label_col_name
        self.label_col_val = self.df[label_col_name].tolist()

    @staticmethod
    def string_accuracy(label: str, prediction: str):
        """
        Compute the accuracy between two strings.
        :param label:
        :param prediction:
        :return: Accuracy Value of
        """
        # check if not string
        if type(prediction) is not str:
            error_mes = f'{prediction} is not a string'
            logger.error(error_mes)
            raise ValueError(error_mes)

        # check if identical
        elif label == prediction:
            return 1

        # element to element comparison
        num_matches = 0
        for c1, c2 in zip_longest(label, prediction):

            # add one if identical
            if c1 == c2:
                num_matches += 1

        return num_matches / max(len(label), len(prediction))

    def column_accuracy(self, pred_list: list):
        """
        Calculate the accuracy between 2 lists of equal length.
        :param pred_list: List of predictions
        :return: Average accuracy of the lists
        """

        # check if not list
        if (type(self.label_col_val) or type(pred_list)) is not list:
            logger.error('Only lists are allowed')
            raise TypeError('Only lists are allowed')

        # if lengths are equal
        elif len(self.label_col_val) != len(pred_list):
            logger.error('Lists are not the same length')
            raise ValueError('Lists are not the same length')

            # accuracy between strings as lists
        acc_list = [
            Accuracy.string_accuracy(label, predict)
            for label, predict in zip(self.label_col_val, pred_list)
        ]

        # return the mean of all string's accuracy
        return statistics.mean(acc_list)

    def accuracy(self, pred_col_names: list):
        """
        Collect the list of all accuracies of each of the column names.

        :param pred_col_names: List of column names to compare with the labels
        :return: List of all specified column's accuracy
        """
        res_list = []
        for col_name in pred_col_names:

            # convert column to list
            col_val = self.df[col_name].tolist()

            # accuracy of that label and prediction columns
            column_acc = Accuracy.column_accuracy(self, col_val)

            # append to list
            res_list.append(column_acc)

        return res_list

# todo implement accuracy in main program
# todo start a master file of all optimized files

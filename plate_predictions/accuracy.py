import logging

logger = logging.getLogger(__name__)

def str_compare(str1: str, str2: str):
    """
    Compares two strings, and returns a value between 0 and 1 for accuracy.
    """
    from itertools import zip_longest

    if type(str2) is not str:
        return 0
    elif str1 == str2:
        return 1

    # loop that checks each value in strings
    num_matches = 0
    for c1, c2 in zip_longest(str1, str2):
        if c1 == c2:
            num_matches += 1
    return num_matches / len(str2)

def column_acc_mean(act_list: list, pred_list: list):
    """
    Compares 2 lists of equal length. Return the mean of the column.
    """
    import statistics
    # check if not list
    if (type(act_list) or type(pred_list)) is not list:
        logger.error('Only lists are allowed')
        raise TypeError('Only lists are allowed')

    # if lengths are equal
    elif len(act_list) != len(pred_list):
        logger.error('Lists are not the same length')
        raise ValueError('Lists are not the same length')

    acc_list = []
    for actual, predict in zip(act_list, pred_list):
        acc_list.append(str_compare(actual, predict))

    mean_acc = statistics.mean(acc_list)  # take the mean of the column's accuracy
    return mean_acc

# todo remove this function and replace with accuracy class
def df_cycle(label_val_list, col_name_list):
    """
    Use DataFrame column names to cycle
    through each column and return the accuracy
    """
    mean_acc_list = []
    for val in col_name_list:
        logging.debug(len(val))
        logging.debug(len(label_val_list))
        # fixme shorten this try except block
        try:
            col_acc = round(column_acc_mean(label_val_list, val) * 100, 2)
            mean_acc_list.append(col_acc)
        except KeyError:
            error_mes = f'Key: {val} not found'
            logger.error(error_mes)
            raise KeyError(error_mes)
    return mean_acc_list

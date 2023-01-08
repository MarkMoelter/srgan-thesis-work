import os

import constants as c
import helpers
from multi_runner import MultiSizeRunner


def combinations(
        low_res_list,
        high_res_list,
        scale_factors
) -> list[tuple]:
    """
    Generate size combination tuples.

    :return: List of tuples ordered: (low, high)
    """
    combos = []
    for lr in low_res_list:
        for hr in high_res_list:
            if (hr // lr) not in scale_factors:
                continue
            combos.append((lr, hr))

    logger.debug(f'Combinations: {combos}')
    return combos


def multi_input_gan():
    # enables relative paths

    # logging init
    logger.info('Starting main()')

    # generate tuples of combinations and scale factors desired
    combo_tuples = combinations(c.LR_TO_TEST, c.HR_TO_TEST, c.SCALE_FACTOR)

    ### Deep Learning Model ###
    # create multi model object
    multi = MultiSizeRunner(
        combo_tuples,
        c.PARAM_PATH,
        c.GEN_PARAM_FILENAME,
        c.DISC_PARAM_FILENAME,
        blur_radius=5)

    # training
    train_df = multi.training(c.TRAIN_DATA_PATH, c.EPOCH_NUM, c.TRAIN_CSV_PATH)
    logger.info(train_df.head())

    # validation
    val_df = multi.validation(c.VAL_DATA_PATH, c.IMG_DIR, c.VAL_CSV_PATH)
    logger.info(val_df.head())

    helpers.tone()


if __name__ == '__main__':
    os.chdir(r'C:\Users\marke\PycharmProjects\Thesis SRGAN')

    logger = helpers.log_init()
    multi_input_gan()

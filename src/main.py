import pandas as pd

import constants as c
import helpers
import recognition as recog
from accuracy import accuracy
from data_gen import generate_dataset
from gan import MultiSizeRunner


def main():
    logger.info('Starting main program')

    GENERATE_DATA = False
    TRAIN_MODEL = False
    VALIDATE_MODEL = False
    RECOGNITION = False
    CHECK_ACCURACY = False

    multi = MultiSizeRunner(
        helpers.combo_tuples(c.LR_TO_TEST, c.HR_TO_TEST, c.SCALE_FACTOR),
        c.PARAM_PATH,
        c.GEN_PARAM_FILENAME,
        c.DISC_PARAM_FILENAME,
        blur_radius=5)

    if GENERATE_DATA:
        generate_dataset(c.SETS, c.MASTER_DATASET_DIR, c.LABEL_PATH)
    if TRAIN_MODEL:
        logger.info(multi.training(c.TRAIN_DATA_PATH, c.EPOCH_NUM).head())
    if VALIDATE_MODEL:
        logger.info(multi.validation(c.VAL_DATA_PATH, c.IMG_DIR).head())
    if RECOGNITION:
        result_path = 'Results/CSVs/recog&acc/copy_Master_Dataset.csv'

        df = pd.read_csv(result_path)
        df = recog.recognition(df)
        df.to_csv(result_path)
    if CHECK_ACCURACY:
        plt_csv = 'Results/CSVs/recog&acc/copy_Master_Dataset.csv'
        combos_as_strings = [
            f'{lr}-{sr}'
            for lr, sr in helpers.combo_tuples(c.LR_TO_TEST, c.HR_TO_TEST)
        ]

        df = pd.read_csv(plt_csv)
        df = accuracy(df, combos_as_strings)
        df.to_csv(c.ACC_CSV_PATH, index=False)

    helpers.tone()
    logger.info('Finished main program')


if '__main__' == __name__:
    logger = helpers.log_init()
    main()

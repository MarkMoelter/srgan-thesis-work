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
        helpers.combos(c.LR_TO_TEST, c.HR_TO_TEST, c.SCALE_FACTOR),
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
        df = pd.read_csv(c.PLATE_CSV)
        df = recog.recognition(df)
        df.to_csv(c.PLATE_CSV)
    if CHECK_ACCURACY:
        df = pd.read_csv(c.PLATE_CSV)

        benchmark_accuracy = accuracy(df, helpers.benchmark_headers(c.HR_TO_TEST))
        lr_accuracy = accuracy(df, helpers.lr_headers(c.LR_TO_TEST))
        sr_accuracy = accuracy(df, helpers.sr_headers(c.LR_TO_TEST, c.HR_TO_TEST))

    helpers.tone()
    logger.info('Finished main program')


if '__main__' == __name__:
    logger = helpers.log_init()
    main()

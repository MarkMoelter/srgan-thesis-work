import logging

import pandas as pd

import constants as c
import utils.helpers as h
from utils.header import Header
import recognition as recog
from accuracy import accuracy
from data_gen import generate_dataset
from gan import MultiSizeRunner


# TODO: REPLACE ALL USES OF DATAFRAME HEADERS WITH HEADER OBJECT AND METHODS!!!
def main():
    logging.info('Starting main program')

    GENERATE_DATA = False
    TRAIN_MODEL = False
    VALIDATE_MODEL = False
    RECOGNITION = False
    CHECK_ACCURACY = False

    multi = MultiSizeRunner(
        h.combos(c.LR_TO_TEST, c.HR_TO_TEST, c.SCALE_FACTOR),
        c.PARAM_PATH,
        c.GEN_PARAM_FILENAME,
        c.DISC_PARAM_FILENAME,
        blur_radius=5)

    if GENERATE_DATA:
        generate_dataset(c.SETS, c.MASTER_DATASET_DIR, c.LABEL_PATH)
    if TRAIN_MODEL:
        logging.info(multi.training(c.TRAIN_DATA_PATH, c.EPOCH_NUM).head())
    if VALIDATE_MODEL:
        logging.info(multi.validation(c.VAL_DATA_PATH, c.IMG_DIR).head())
    if RECOGNITION:
        df = pd.read_csv(c.PLATE_CSV)
        df = recog.recognition(df)
        df.to_csv(c.PLATE_CSV)
    if CHECK_ACCURACY:
        df = pd.read_csv(c.PLATE_CSV)

        benchmark_accuracy = accuracy(df, Header().benchmark_headers(c.HR_TO_TEST))
        lr_accuracy = accuracy(df, Header().lr_headers(c.LR_TO_TEST))
        sr_accuracy = accuracy(df, Header().sr_headers(c.LR_TO_TEST, c.HR_TO_TEST))

        logging.info(benchmark_accuracy)
        logging.info(lr_accuracy)
        logging.info(sr_accuracy)

    h.tone()
    logging.info('Finished main program')


if '__main__' == __name__:
    main()

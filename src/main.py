import pandas as pd

import constants as c
import helpers
import recognition as recog
from accuracy import accuracy
from data_gen import generate_dataset
from gan import MultiSizeRunner


def format_sizes() -> list[str]:
    """Format the combinations from combo_tuples function. Specifically for the accuracy dataframes."""
    return [
        f'{lr}-{sr}'
        for lr, sr in helpers.combo_tuples(c.LR_TO_TEST, c.HR_TO_TEST)
    ]


# accuracy
def accuracy_main():
    # TODO: Refactor this
    plt_csv = 'Results/CSVs/recog&acc/copy_Master_Dataset.csv'

    plate_df = pd.read_csv(plt_csv)
    labels = [plate_df['Real Plates'].tolist()]

    # benchmark accuracy
    for num in c.HR_TO_TEST:
        predictions = plate_df[f'{num}_Benchmark'].tolist()
        logger.info(
            f"{num}_Benchmark Accuracy: "
            f"{accuracy(plate_df, labels, predictions)}"
        )

    # lr accuracy
    lr_col_list = [
        f'LR:{val}'
        for val in c.LR_TO_TEST
    ]  # list of column names
    # nested lists of predicted plates
    lr_acc_test_plts = [
        plate_df[f'{val}'].tolist()
        for val in lr_col_list
    ]

    lr_size = accuracy(plate_df, labels, lr_acc_test_plts)
    logger.info(f'LR Acc: {lr_size}')

    # sr accuracy
    sr_col_list = [f'SR:{val}' for val in format_sizes()]
    sr_acc_test_plts = [plate_df[f'{val}'].tolist() for val in sr_col_list]
    sr_size = accuracy(plate_df, labels, sr_acc_test_plts)
    logger.info(f'SR Acc: {sr_size}')  # log

    # convert to dataframe and save as csv
    acc_df = pd.DataFrame(data=[lr_size, sr_size], index=['LR', 'SR'], columns=format_sizes())
    logger.info(acc_df.head())

    acc_df.to_csv(c.ACC_CSV_PATH, index=False)


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
        accuracy_main()

    helpers.tone()
    logger.info('Finished main program')


if '__main__' == __name__:
    logger = helpers.log_init()
    main()

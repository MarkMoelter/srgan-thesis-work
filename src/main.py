import pandas as pd

from accuracy import accuracy
import constants as c
import helpers
from recognition import Recognition


# plate recognition
def recognition_main(plate_df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses OCR to recognize the high resolution and low resolution plates.

    :param plate_df: The dataframe containing the plate labels.
    """
    # TODO: Move this somewhere else
    # collect the OCR predictions
    recog = Recognition(plate_df)
    recog.benchmark_recognition()
    recog.low_resolution_recognition()
    recog.high_resolution_recognition()

    logger.info('Plate predictions finished')
    return recog.df


# accuracy
def accuracy_main():
    # TODO: Refactor this
    lr_sr_size_list = [
        f'{lr}-{sr}'
        for lr, sr in helpers.combo_tuples(c.LR_TO_TEST, c.HR_TO_TEST)
    ]

    plt_csv = 'Results/CSVs/recog&acc/copy_Master_Dataset.csv'

    plate_df = pd.read_csv(plt_csv)
    labels = [plate_df['Real Plates'].tolist()]

    # benchmark accuracy
    for num in c.HR_TO_TEST:
        predictions = plate_df[f'{num}_Benchmark'].tolist()
        logger.info(
            f"{num}_Benchmark Accuracy: "
            f"{accuracy(plate_df, labels ,predictions)}"
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
    sr_col_list = [f'SR:{val}' for val in lr_sr_size_list]
    sr_acc_test_plts = [plate_df[f'{val}'].tolist() for val in sr_col_list]
    sr_size = accuracy(plate_df, labels, sr_acc_test_plts)
    logger.info(f'SR Acc: {sr_size}')  # log

    # convert to dataframe and save as csv
    acc_df = pd.DataFrame(data=[lr_size, sr_size], index=['LR', 'SR'], columns=lr_sr_size_list)
    logger.info(acc_df.head())

    acc_df.to_csv(c.ACC_CSV_PATH, index=False)


def main():
    from data_gen import generate_dataset
    from gan import MultiSizeRunner

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
        df = recognition_main(df)
        df.to_csv(result_path)
    if CHECK_ACCURACY:
        accuracy_main()

    helpers.tone()
    logger.info('Finished main program')


if '__main__' == __name__:
    logger = helpers.log_init()
    main()

import pandas as pd

import constants as c
import helpers
import accuracy
from gan import MultiSizeRunner


# plate recognition
def recognition_main():
    lr_list = [16, 32, 64]
    sr_list = [64, 128, 256, 512]
    exceptions = ['16-256', '16-512', '32-512', '64-64']

    csv_pth = 'Results/CSVs/recog&acc/copy_Master_Dataset.csv'
    plate_df = pd.read_csv(csv_pth)

    for h_res in [64, 128, 256]:
        img_path = f'Results/Images/{h_res}'
        plate_df[f'{h_res}_Benchmark'] = accuracy.plate_reader(img_path, im_show=False)
        plate_df.to_csv(csv_pth, index=False)

    benchmark = 'Results/master_dataset/testing_set'
    plate_df['512_Benchmark'] = accuracy.plate_reader(benchmark, im_show=False)

    logger.debug(plate_df.head())
    plate_df.to_csv(csv_pth, index=False)

    # all SR combos
    for LR in lr_list:
        for SR in sr_list:
            if f'{LR}-{SR}' in exceptions:
                continue

            # LR images
            LR_img_path = f'Results/Images/{LR}to{SR}/LR'
            pred = accuracy.plate_reader(LR_img_path, im_show=False)
            plate_df[f'LR:{LR}'] = pred
            plate_df.to_csv(csv_pth, index=False)
            logger.info(f'{LR} LR Predictions saved')

            # SR images
            SR_img_path = f'Results/Images/{LR}to{SR}/SR'
            pred = accuracy.plate_reader(SR_img_path, im_show=False)
            plate_df[f'SR:{LR}-{SR}'] = pred
            plate_df.to_csv(csv_pth, index=False)
            logger.info(f'{SR} SR Predictions saved')


# accuracy
def accuracy_main():
    lr = [16, 32, 64]
    sr = [64, 128, 256, 512]
    exceptions = ['16-256', '16-512', '32-512', '64-64']
    lr_sr_size_list = helpers.combo_strings(lr, sr, exceptions)

    plt_csv = 'Results/CSVs/recog&acc/copy_Master_Dataset.csv'
    predicted_plates = pd.read_csv(plt_csv)  # read csv

    # plate labels
    real_plates = predicted_plates[f'Real Plates'].tolist()

    # benchmark accuracy
    for num in sr:
        logger.info(
            f"{num}_Benchmark Accuracy: "
            f"{accuracy.df_cycle(real_plates, [predicted_plates[f'{num}_Benchmark'].tolist()])}"
        )

    # lr accuracy
    lr_col_list = [f'LR:{val}' for val in lr]  # list of column names
    # nested lists of predicted plates
    lr_acc_test_plts = [predicted_plates[f'{val}'].tolist() for val in lr_col_list]

    lr_size = accuracy.df_cycle(real_plates, lr_acc_test_plts)
    logger.info(f'LR Acc: {lr_size}')  # log

    # sr accuracy
    sr_col_list = [f'SR:{val}' for val in lr_sr_size_list]  # list of column names
    # nested lists of predicted plates
    sr_acc_test_plts = [predicted_plates[f'{val}'].tolist() for val in sr_col_list]

    sr_size = accuracy.df_cycle(real_plates, sr_acc_test_plts)
    logger.info(f'SR Acc: {sr_size}')  # log

    # convert to dataframe and save as csv
    acc_df = pd.DataFrame(data=[lr_size, sr_size], index=['LR', 'SR'], columns=lr_sr_size_list)
    logger.info(acc_df.head())

    acc_df.to_csv(c.ACC_CSV_PATH, index=False)


def main():
    logger.info('Starting main()')
    # generate tuples of combinations and scale factors desired
    combos_to_test = helpers.combo_tuples(c.LR_TO_TEST, c.HR_TO_TEST, c.SCALE_FACTOR)

    ### Deep Learning Model ###
    # create multi model object
    multi = MultiSizeRunner(
        combos_to_test,
        c.PARAM_PATH,
        c.GEN_PARAM_FILENAME,
        c.DISC_PARAM_FILENAME,
        blur_radius=5)

    # training
    train_df = multi.training(c.TRAIN_DATA_PATH, c.EPOCH_NUM)
    logger.info(train_df.head())

    # validation
    val_df = multi.validation(c.VAL_DATA_PATH, c.IMG_DIR)
    logger.info(val_df.head())


if '__main__' == __name__:
    # enables relative path
    # os.chdir(r'C:\Users\marke\PycharmProjects\Thesis SRGAN')

    # logging init
    logger = helpers.log_init()

    # program call
    main()

    # tone to indicate program finished
    helpers.tone()

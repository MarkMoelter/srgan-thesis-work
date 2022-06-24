import os

import numpy
import pandas as pd

import constants as c
import data_gen
import helpers
import plate_predictions
from multi_size_runner import MultiSizeRunner


# todo check if working:
#   generation - Done
#   training - Done
#   validation
#   recognition
#   accuracy

# todo remove unnecessary files:
#   GAN.model.train
#   GAN.model.validation
#   multi_train
#   multi_validation

# todo new features:
#   get metrics working

class DatasetInfo:
    """Dataset properties in one place"""
    def __init__(self, low_res_list, high_res_list, scale_factors):
        self.LR = low_res_list
        self.HR = high_res_list
        self.scale_factors = scale_factors

    def combinations(self):
        """
        Generate size combinations for implementation in multi size testing
        :return: Tuples with (low, high)
        """
        combos = []
        for lr in self.LR:
            for hr in self.HR:
                if (hr//lr) not in self.scale_factors:
                    continue
                combos.append((lr, hr))

        logger.debug(f'Combinations: {combos}')
        return combos

    def print(self):
        print(self.combinations())

class Main:
    def __init__(
            self,
            combinations,
            dataset_path,
            dataset_csv_path,
    ):
        # combinations
        self.combinations = combinations

        self.dataset_path = dataset_path
        self.dataset_csv_path = dataset_csv_path


    def dataset_generation(self, sets):
        """
        Generate a number of license plates
        :param sets: The number of 252 plates to generate
        :return: Dataframe of Plates
        """
        return data_gen.plate_gen(
            sets,
            img_save_dir=self.dataset_path,
            label_csv_path=self.dataset_csv_path
        )

    def recognition(self):
        pass

    def accuracy(self):
        pass


# plate recognition
def recognition_main():
    lr_list = [16, 32, 64]
    sr_list = [64, 128, 256, 512]
    exceptions = ['16-256', '16-512', '32-512', '64-64']

    csv_pth = 'Results/CSVs/recog&acc/copy_Master_Dataset.csv'
    plate_df = pd.read_csv(csv_pth)

    for h_res in [64, 128, 256]:
        img_path = f'Results/Images/{h_res}'
        plate_df[f'{h_res}_Benchmark'] = plate_predictions.plate_reader(img_path, im_show=False)
        plate_df.to_csv(csv_pth, index=False)

    benchmark = 'Results/dataset/testing_set'
    plate_df['512_Benchmark'] = plate_predictions.plate_reader(benchmark, im_show=False)

    logger.debug(plate_df.head())
    plate_df.to_csv(csv_pth, index=False)

    # all SR combos
    for LR in lr_list:
        for SR in sr_list:
            if f'{LR}-{SR}' in exceptions:
                continue

            # LR images
            LR_img_path = f'Results/Images/{LR}to{SR}/LR'
            pred = plate_predictions.plate_reader(LR_img_path, im_show=False)
            plate_df[f'LR:{LR}'] = pred
            plate_df.to_csv(csv_pth, index=False)
            logger.info(f'{LR} LR Predictions saved')

            # SR images
            SR_img_path = f'Results/Images/{LR}to{SR}/SR'
            pred = plate_predictions.plate_reader(SR_img_path, im_show=False)
            plate_df[f'SR:{LR}-{SR}'] = pred
            plate_df.to_csv(csv_pth, index=False)
            logger.info(f'{SR} SR Predictions saved')

# accuracy
# todo refactor accuracy
def accuracy_main():
    lr = [16, 32, 64]
    sr = [64, 128, 256, 512]
    exceptions = ['16-256', '16-512', '32-512', '64-64']
    lr_sr_size_list = helpers.combo_gen(lr, sr, exceptions)

    plt_csv = 'Results/CSVs/recog&acc/copy_Master_Dataset.csv'
    predicted_plates = pd.read_csv(plt_csv)  # read csv

    # plate labels
    real_plates = predicted_plates[f'Real Plates'].tolist()

    # benchmark accuracy
    for num in sr:
        logger.info(
            f"{num}_Benchmark Accuracy: "
            f"{plate_predictions.df_cycle(real_plates, [predicted_plates[f'{num}_Benchmark'].tolist()])}"
        )

    # lr accuracy
    lr_col_list = [f'LR:{val}' for val in lr]  # list of column names
    # nested lists of predicted plates
    lr_acc_test_plts = [predicted_plates[f'{val}'].tolist() for val in lr_col_list]

    lr_size = plate_predictions.df_cycle(real_plates, lr_acc_test_plts)
    logger.info(f'LR Acc: {lr_size}')  # log

    # sr accuracy
    sr_col_list = [f'SR:{val}' for val in lr_sr_size_list]  # list of column names
    # nested lists of predicted plates
    sr_acc_test_plts = [predicted_plates[f'{val}'].tolist() for val in sr_col_list]

    sr_size = plate_predictions.df_cycle(real_plates, sr_acc_test_plts)
    logger.info(f'SR Acc: {sr_size}')  # log

    # convert to dataframe and save as csv
    acc_df = pd.DataFrame(data=[lr_size, sr_size], index=['LR', 'SR'], columns=lr_sr_size_list)
    logger.info(acc_df.head())

    csv_out = 'Results/CSVs/recog&acc/Final_Accuracy.csv'
    acc_df.to_csv(csv_out, index=False)


######################################
# Background removed accuracy and recognition
def BG_removed_RnA():
    image_pth = f'Results/Processed'
    out_pth = f'Results/CSVs/recog&acc/BackRemove_test.csv'
    logger.info('Starting function')
    df = pd.read_csv(out_pth)

    df['Removed'] = plate_predictions.plate_reader(image_pth, im_show=False)
    df.to_csv(out_pth, index=False)
    logger.info(df.head())

    real_plate = [df['Real Plates'].tolist()]
    pred_plate = [df['Removed'].tolist()]

    logger.info(f'BG Removed Acc: {plate_predictions.df_cycle(real_plate, pred_plate)}')

    logger.info('Finished function')


def other_benchmark():
    """To make calling easier? inner func resizes images"""
    from tqdm import tqdm
    import cv2
    from PIL import Image
    from torchvision.transforms import Resize

    image_dir = 'Results/dataset/testing_set'

    def hr_bench_save(size: list, save_dir: str):
        """Resize dataset images to specified sizes"""
        for h_res in size:

            _dir = os.listdir(image_dir)
            bar = tqdm(_dir, desc='Cycling through benchmarks')
            # loop through all files in a dir
            for filename in bar:

                # skip iteration if file does not exist
                if not os.path.isfile(os.path.join(image_dir, filename)):
                    continue

                # open the plate image with cv2
                plt_pth = f'{image_dir}/{filename}'
                img = cv2.imread(plt_pth)  # open img
                bar.set_description(f'Resizing dataset to {h_res}')

                # new addition, convert to HR size then blur
                def resize(cv_img, size):
                    cv_RGB_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)  # grayscale
                    pil_img = Image.fromarray(cv_RGB_img)
                    hr_img = Resize(size)(pil_img)
                    cv_resized = numpy.array(hr_img)
                    return cv2.cvtColor(cv_resized, cv2.COLOR_RGB2BGR)

                # resize images
                img = resize(img, h_res)

                # save images
                image_save_path = f'{save_dir}/{h_res}'
                helpers.create_pth(image_save_path)
                cv2.imwrite(f'{image_save_path}/{filename}', img)

    hr_bench_save(size=[64], save_dir='Results/Images')
    logger.info('Finished Saving the benchmark images')


def main():
    logger.info('Starting main()')
    # generate tuples of combinations and scale factors desired
    combo_tuples = DatasetInfo(c.LR_LIST, c.HR_LIST, c.SCALE_TUPLE).combinations()

    ### Deep Learning Model ###
    # create multi model object
    parameter_path = 'Results/params'
    multi = MultiSizeRunner(combo_tuples, parameter_path, blur_radius=5)

    # training
    # train_dataset_path = 'Results/dataset/training_set/BlurRad_0'
    # train_df = multi.training(train_dataset_path, 1)
    # logger.info(train_df.head())

    # validation
    val_data_path = 'Results/dataset/testing_set'
    val_df = multi.validation(val_data_path, 'Results/Images')
    logger.info(val_df.head())





if '__main__' == __name__:
    # enables relative paths
    # os.chdir(r'C:\Users\marke\PycharmProjects\Thesis SRGAN')

    # logging init
    logger = helpers.log_init()

    # program call
    main()

    # tone to indicate program finished
    helpers.tone()

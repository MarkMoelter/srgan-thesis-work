import os

import pandas as pd
from tqdm import tqdm

from plate_generator import PlateGenerator
from data_gen.plate_templates.rectangular_plate import RectangularPlate


def dataset_generate():
    SETS = 6
    IMG_DIR = 'data_gen/master_dataset'
    CSV_PATH = 'data_gen/image_labels.csv'

    # plate_dir = 'Results/master_dataset/testing_set'
    # master_csv = f'Results/CSVs/Master_Dataset.csv'

    gen = PlateGenerator(RectangularPlate)
    plates = gen.multi_plates(SETS)

    plate_list = [
        [plate.filename, plate.label]
        for plate in plates
    ]

    # dataframe creation and save
    dataframe = pd.DataFrame(plate_list, columns=['Filename', 'Labels'])
    dataframe.to_csv(CSV_PATH, index=False)

    # save the images
    for plate in tqdm(plates):
        plate.save_plate(IMG_DIR)


if __name__ == '__main__':
    os.chdir(r'C:\Users\marke\PycharmProjects\Thesis SRGAN')

    dataset_generate()

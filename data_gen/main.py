import os
import pandas as pd

from generator import Generator
from template import Template


def main():
    SETS = 6
    IMG_DIR = ''
    CSV_PATH = ''

    # plate_dir = 'Results/dataset/testing_set'
    # master_csv = f'Results/CSVs/Master_Dataset.csv'

    gen = Generator(Template)
    plates = gen.multi_plates(SETS)

    plate_list = [
        [plate.filename, plate.label]
        for plate in plates
    ]

    # dataframe creation and save
    dataframe = pd.DataFrame(plate_list, columns=['Filename', 'Labels'])
    print(dataframe.head())
    # dataframe.to_csv(CSV_PATH, index=False)

    # save the images
    # for plate in plates:
    #     plate.save_plate(IMG_DIR)


if __name__ == '__main__':
    os.chdir(r'C:\Users\marke\PycharmProjects\Thesis SRGAN')

    main()

import pandas as pd
from tqdm import tqdm

from .plate_generator import PlateGenerator
from .plate_params import RectangularPlate


def generate_dataset(sets: int, image_directory: str, label_path: str) -> None:
    gen = PlateGenerator(RectangularPlate)
    plates = gen.generate_plate_sets(sets)

    plate_list = [
        [plate.filename, plate.label]
        for plate in plates
    ]

    # dataframe creation and save
    dataframe = pd.DataFrame(plate_list, columns=['Filename', 'Labels'])
    dataframe.to_csv(label_path, index=False)

    # save the images
    for plate in tqdm(plates):
        plate.save_plate(image_directory)

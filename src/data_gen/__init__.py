def generate_dataset(sets: int, image_directory: str, label_csv_path: str) -> None:
    """
    Generates a dataset of labeled license plates.
    Saves a CSV containing the dataset labels to the specified path.

    :param sets: The number of image sets to generate.
    :param image_directory: The directory to save the dataset to.
    :param label_csv_path: The path with filename to save the label csv to.
    """
    import pandas as pd
    from tqdm import tqdm

    from .plate_generator import PlateGenerator
    from .plate_properties import RectangularPlate
    from .label_properties import PlateLabel

    gen = PlateGenerator(RectangularPlate, PlateLabel)
    plates = gen.generate_sets(sets)

    plate_list = [
        [plate.filename, plate.label]
        for plate in plates
    ]

    # dataframe creation and save
    dataframe = pd.DataFrame(plate_list, columns=['Filename', 'Labels'])
    dataframe.to_csv(label_csv_path, index=False)

    # save the images
    for plate in tqdm(plates):
        plate.save_plate(image_directory)

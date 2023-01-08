import logging
import os
from tqdm import tqdm
import cv2
from PIL import Image

import pandas as pd
import pytesseract as tess
import torchvision.transforms as transforms

import constants as c
from plate import Plate

logger = logging.getLogger(__name__)


# TODO: Get rid of plate_reader from main.py
class PlateReader:
    def __init__(
            self,
            image_dir: str,
            label_csv_path: str = None,
            prediction_csv_path: str = None,
            image_show=False
    ):
        self.image_dir = image_dir
        self.label_csv_path = label_csv_path
        self.prediction_csv_path = prediction_csv_path
        self.image_show = image_show
        self.plate_obj_list = []

    def reader_loop(self):
        """
        Read the characters from the plates and store them in a Dataframe.
        """
        # set pytesseract as environment var
        tess.pytesseract.tesseract_cmd = c.PYTESSERACT_ENV_VAR_PATH

        bar = tqdm(os.listdir(self.image_dir), desc='Predicting plates')

        # loop through all files in a dir
        for filename in bar:

            # skip iteration if file does not exist
            if not os.path.isfile(os.path.join(self.image_dir, filename)):
                continue

            bar.set_description(f'Predicting plate: {filename}')

            plate_path = f'{self.image_dir}/{filename}'
            transformed_plate = self._transform(plate_path)

            if self.image_show:
                transformed_plate.show()

            # generate the prediction
            prediction = tess.image_to_string(
                transformed_plate,
                lang='eng',
                config='--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            )

            # filter
            blacklist_chars = ['-', ' ']
            filter_prediction = self._filter_string(prediction, blacklist_chars)

            # add to the plate list
            self.plate_obj_list.append(Plate(filename, filename, filter_prediction))

        # save a csv
        if self.prediction_csv_path is not None:
            self._save_csv()

        return [plate.prediction for plate in self.plate_obj_list]

    @staticmethod
    def _filter_string(prediction: str, blacklist: list[str]):
        """
        Filters characters out of a prediction string.

        :param prediction: The string to filter.
        :param blacklist: Characters to be filtered out.
        :return: The filtered prediction
        """

        for char in blacklist:
            prediction = prediction.replace(char, '')

        return prediction

    @staticmethod
    def _transform(img_path: str) -> Image:
        """
        Convert an image to grayscale, center crop and resize it.

        :param img_path: path to image to transform
        :return: Transformed image as a PIL Image
        """

        # open image as cv2 image
        img = cv2.imread(img_path)

        # grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # blur slightly to denoise
        blur_radius = 1
        img = cv2.GaussianBlur(img, (blur_radius, blur_radius), 0)

        # center crop params
        img_height, img_width = img.shape
        crop_h = int(img_height * 280 / 512)
        crop_w = int(img_width * 450 / 512)

        # resize params
        resize_h = 256
        resize_w = resize_h * 4

        # apply the transforms
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop((crop_h, crop_w)),
            transforms.Resize(
                (resize_h, resize_w),
                interpolation=transforms.InterpolationMode.BICUBIC
            )
        ])(img)

    def _save_csv(self):
        """Save plates to csv."""
        pd.DataFrame(
            data=[plate.prediction for plate in self.plate_obj_list]
        ).to_csv(self.prediction_csv_path, index=False)

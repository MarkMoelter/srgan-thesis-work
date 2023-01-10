import logging
import os

import cv2
import numpy as np
import pandas as pd
import pytesseract as tess
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

import constants as c
from plate import Plate

logger = logging.getLogger(__name__)


class PlateReader:
    def __init__(self, image_dir: str, prediction_csv_path: str = None):
        self.image_dir = image_dir
        self.prediction_csv_path = prediction_csv_path

    def predictions(self, image_show: bool = False):
        """
        Read the characters from the plates and store them in a Dataframe.
        """
        # set pytesseract as environment var
        tess.pytesseract.tesseract_cmd = c.PYTESSERACT_ENV_VAR_PATH

        bar = tqdm(os.listdir(self.image_dir), desc='Predicting plates')
        plates = []
        # loop through all files in a dir
        for filename in bar:
            path = os.path.join(self.image_dir, filename)

            # skip iteration if file does not exist
            if not os.path.isfile(path):
                continue

            bar.set_description(f'Predicting plate: {filename}')

            plate_img = cv2.imread(path)
            transformed_plate: Image = self._transform_image(plate_img)

            if image_show:
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
            plates.append(Plate(filename, filename, filter_prediction))

        # save a csv
        if self.prediction_csv_path is not None:
            self._save_csv(plates)

        return [plate.prediction for plate in plates]

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
    def _transform_image(img: np.ndarray) -> Image:
        """
        Convert a cv2 image to grayscale, center crop, resize it,
        and return it as a PIL Image.

        :param img: cv2 image to be transformed.
        :return: Transformed image as a PIL Image.
        """
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

    def _save_csv(self, plates: list[Plate]) -> None:
        """
        Save plates to csv.

        :param plates: List of plate objects.
        """
        pd.DataFrame(
            data=[plate.prediction for plate in plates]
        ).to_csv(self.prediction_csv_path, index=False)

import logging
import os

import cv2
import pandas as pd
import pytesseract as tess
import torchvision.transforms as transforms
from tqdm import tqdm

logger = logging.getLogger(__name__)
# todo annotate plate reader
def plate_reader(
        image_dir: str,
        csv_in_path: str = None,
        csv_out_path: str = None,
        im_show=False
):
    # set pytesseract as environment var
    tess.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'


    _dir = os.listdir(image_dir)
    plate_dict = {
        'Plate Filename': [],
        'Real Plate': [],
        'Predicted Plate': []
    }

    bar = tqdm(_dir, desc='Predicting plates')
    # loop through all files in a dir
    for filename in bar:

        # skip iteration if file does not exist
        if not os.path.isfile(os.path.join(image_dir, filename)):
            continue

        # open the plate image with cv2
        plt_pth = f'{image_dir}/{filename}'
        img = cv2.imread(plt_pth)  # open img
        bar.set_description(f'Predicting plate size: {img.shape[0]}')

        # logger.debug('shape when opened %s', img.shape)

        # transforms
        rad = 1
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # grayscale
        img = cv2.GaussianBlur(img, (rad, rad), 0)  # blur to denoise
        h, w = img.shape
        crop_h, crop_w = int(h * 280 / 512), int(w * 450 / 512)
        resize_h = 256
        resize_w = resize_h*4
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop((crop_h, crop_w)),
            transforms.Resize(
                (resize_h, resize_w),
                interpolation=transforms.InterpolationMode.BICUBIC
            )
        ])
        preprocessed_img = transform(img)

        # shows each image
        if im_show:
            preprocessed_img.show()

        # image to text, filter spaces out
        predicted_result = tess.image_to_string(
            preprocessed_img,
            lang='eng',
            config='--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        )

        # filter text
        filter_predicted_result = ''.join(predicted_result.split()).replace('-', '').replace(' ', '')
        plate_dict['Predicted Plate'].append(filter_predicted_result)
        # logger.debug(plate_dict['Predicted Plate'][-1])

        # combine individual labels into plates, only for master dataset
        if csv_in_path is not None:
            # read the csv with plate file names and labels
            csv = pd.read_csv(csv_in_path)
            # remove extension and join characters into one string
            filename_wo_ext = os.path.splitext(filename)[0]
            plate_chars = ''.join([csv.iloc[int(filename_wo_ext), j] for j in range(1, len(csv.axes[1]))])
            plate_dict['Plate Filename'].append(filename_wo_ext)
            plate_dict['Real Plate'].append(plate_chars)

    # save predictions as csv
    if csv_out_path is not None:
        data_frame = pd.DataFrame(data=plate_dict['Predicted Plate'])
        data_frame.to_csv(csv_out_path, index=False)

    if csv_in_path is not None:
        return plate_dict['Real Plate'], plate_dict['Predicted Plate']
    else:
        return plate_dict['Predicted Plate']


if '__main__' == __name__:
    os.chdir(r'C:\Users\marke\PycharmProjects\Thesis SRGAN')

    test = 'Results/Images/64to128/LR'

    plate_reader(test, im_show=True)

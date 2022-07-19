import logging
from os import listdir, path

import cv2
import numpy
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, Resize

logger = logging.getLogger(__name__)


def is_image_file(filename):
    """Checks if the file is one of file types listed below"""
    file_ext = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    return (
        any(filename.endswith(extension) for extension in file_ext)
    )


class TrainDataset(Dataset):
    """
    Returns random crop of hr image (crop_size, crop_size).
    Resizes the random crop to (crop_size//upscale_factor).
    crop_size: 64 // upscale_factor: 8; lr image size: (8, 8)\n
    Returns hr: (64, 64) and lr: (8, 8).\n
    Scale factor needs to be maintained at all times.
    Uses a low res image size of 8 or larger, and picks the high res crop size
    according to scale factor.
    """

    def __init__(self, dataset_dir, high_res_size, low_res_size, blur=0):
        super().__init__()
        self.blur = blur
        upscale_factor = high_res_size // low_res_size
        train_low_res_size = max(low_res_size // upscale_factor, 8)

        # transforms
        self.hr_transform = Compose([
            Resize(high_res_size),
            RandomCrop(train_low_res_size * upscale_factor)
        ])
        self.lr_transform = Resize(train_low_res_size)

        # image files
        self.image_filenames = [
            path.join(dataset_dir, x)
            for x in listdir(dataset_dir)
            if is_image_file(x)
        ]
        logger.debug(f'Dataset length: {len(self.image_filenames)}')

    def __getitem__(self, index):
        img = Image.open(self.image_filenames[index])

        hr_image = self.hr_transform(img)
        unblurred_hr = hr_image

        # blur
        if self.blur > 0:
            cv_img = numpy.array(hr_image)
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
            blur_img = cv2.GaussianBlur(cv_img, (self.blur, self.blur), 0)
            blur_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB)
            hr_image = Image.fromarray(blur_img)

        lr_image = self.lr_transform(hr_image)
        return ToTensor()(lr_image), ToTensor()(unblurred_hr)

    def __len__(self):
        return len(self.image_filenames)

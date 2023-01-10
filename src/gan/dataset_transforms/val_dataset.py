import logging
import os

import cv2
import numpy
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor, Resize, InterpolationMode

logger = logging.getLogger(__name__)


def is_image_file(filename):
    """Checks if the file is one of file types listed below"""
    file_ext = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    return (
        any(filename.endswith(extension) for extension in file_ext)
    )


class ValDataset(Dataset):
    def __init__(self, dataset_dir, high_res_size, low_res_size, blur=0):
        super().__init__()
        self.high_res = high_res_size
        self.low_res = low_res_size
        self.blur = blur
        # transforms
        self.scale_down = Resize(self.low_res)
        self.scale_up = Resize(self.high_res, interpolation=InterpolationMode.BICUBIC)

        self.image_filenames = [
            os.path.join(dataset_dir, x)
            for x in os.listdir(dataset_dir)
            if is_image_file(x)
        ]
        logger.debug(f'Dataset length: {len(self.image_filenames)}')

    def __getitem__(self, index):
        """Creates LR, HR, and HR interpolated tensors"""
        # open image
        original_img = Image.open(self.image_filenames[index])

        # resize if not the correct size.
        # always have original be >= to self.high_res
        if original_img.size[0] != self.high_res:
            target = Resize(self.high_res)(original_img)
        else:
            target = original_img

        # save unblurred img for comparisons
        unblurred_target = target

        # blur
        if self.blur > 0:
            a = numpy.array(target)
            b = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
            c = cv2.GaussianBlur(b, (self.blur, self.blur), 0)
            d = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
            target = Image.fromarray(d)

        lr_image = self.scale_down(target)  # scale hr down to lr
        interpolated = self.scale_up(lr_image)  # interpolation

        return ToTensor()(lr_image), ToTensor()(interpolated), ToTensor()(unblurred_target)

    def __len__(self):
        return len(self.image_filenames)

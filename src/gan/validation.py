import logging
import statistics

import torch
import torchvision.utils as utils
from tqdm import tqdm

import utils.path
from .metrics import psnr, ssim
from .model import Generator

logger = logging.getLogger(__name__)


class Validation:
    """
    Validation/Testing of the model.
    Using images the model has never been trained with.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(
            self,
            generator: Generator,
            generator_param_path,
            dataset,
            img_save_dir: str
    ):
        self.generator = generator.to(self.device)
        self.param_path = generator_param_path
        self.dataset = dataset
        self.img_save_dir = img_save_dir

        # metrics
        self.sr_psnr = []
        self.sr_ssim = []
        self.inter_psnr = []
        self.inter_ssim = []

        # images
        self.grid_images = []
        self.low_res_images = []
        self.super_res_images = []

    def main_loop(self) -> None:
        """
        Main loop for running the validation of the model.
        appends the
        """
        images = {
            'grid': [],
            'super_res': [],
            'low_res': [],
        }

        # load and set generator to eval
        self._load_generator()

        val_bar = tqdm(self.dataset)
        with torch.no_grad():
            for low_res, interpolated, reference in val_bar:
                # send images to device
                super_res = self.generator(low_res.to(self.device))
                interpolated = interpolated.to(self.device)
                reference = reference.to(self.device)

                self.sr_psnr.append(psnr(reference, super_res))
                self.sr_ssim.append(ssim(reference, super_res))

                self.inter_psnr.append(psnr(reference, super_res))
                self.inter_ssim.append(ssim(reference, super_res))

                val_bar.set_description(
                    desc=(
                        f"[Converting LR images to SR images] "

                        f"Super PSNR: {psnr(reference, super_res):.4f} dB "
                        f"Super SSIM: {ssim(reference, super_res):.4f} "

                        f"Inter PSNR: {psnr(reference, super_res):.4f} dB "
                        f"Inter SSIM: {ssim(reference, super_res):.4f}"
                    )
                )

                # Images
                grid_list = [
                    reference.data.cpu().squeeze(0),
                    interpolated.cpu().squeeze(0),
                    super_res.data.cpu().squeeze(0)
                ]

                images['grid'].extend(grid_list)
                images['super_res'].append(super_res.data.cpu().squeeze(0))
                images['low_res'].append(low_res.data.cpu().squeeze(0))

            self._save_images(images['grid'], len(val_bar), 'grid', grid=True)
            self._save_images(images['super_res'], len(val_bar), 'SR')
            self._save_images(images['low_res'], len(val_bar), 'LR')

    def collect_metrics(self):
        """Return the mean value of each metric collected in the main loop.

        Order:

        - sr_psnr
        - sr_ssim
        - inter_psnr
        - inter_ssim
        """

        metric_list = [
            self.sr_psnr,
            self.sr_ssim,
            self.inter_psnr,
            self.inter_ssim
        ]

        metric_list = [statistics.mean(metric) for metric in metric_list]
        return metric_list

    def _load_generator(self) -> None:
        """
        Load the generator from the state dictionary created during training.
        Set the generator to evaluation mode.
        """

        self.generator.load_state_dict(torch.load(self.param_path))
        self.generator.eval()

    @staticmethod
    def _prepare_images(images: list, chunk_size: int) -> list[torch.Tensor]:
        """
        Prepare a list of image tensors to be saved.

        :param images: List of image tensors
        :param chunk_size: How many sections to divide the images into.
        :return: List of tensors in the proper chunk size.
        """

        stacked_images = torch.stack(images)
        return torch.chunk(stacked_images, chunk_size)

    def _save_images(
            self,
            images: list,
            chunk_size: int,
            subdirectory: str,
            grid: bool = False,
            image_per_row: int = 3,
    ) -> None:
        """
        Saves the images in the dictionary to the image save path.

        :param images: List containing all the images to save.
        :param chunk_size: How many sections to divide the images into.
        :param subdirectory: The directory to save the images to.
        :param grid: If the resulting images will be a grid of images.
        :param image_per_row: Number of images allowed in each row of the grid.
            Only works if grid param is True.
        """

        filepath = f'{self.img_save_dir}/{subdirectory}'
        utils.path.create_path(filepath)

        prepared_images = self._prepare_images(images, chunk_size)
        for index, image in enumerate(tqdm(prepared_images)):

            if grid:
                image = utils.make_grid(image, nrow=image_per_row)

            image_path = f'{filepath}/{str(index).zfill(4)}.png'
            utils.save_image(image, image_path)

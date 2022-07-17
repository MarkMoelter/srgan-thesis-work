import logging
import statistics

import torch
import torchvision.utils as utils
from tqdm import tqdm

import helpers
from metrics import CalculateMetrics
from model import Generator

logger = logging.getLogger(__name__)


# TODO: replace run_model with model_training and model_validation
class ModelValidation:
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
        self.generator = generator
        self.param_path = generator_param_path
        self.dataset = dataset
        self.img_save_dir = img_save_dir

    def main_loop(self) -> dict:
        """
        Main loop for running the validation of the model.

        :return: Dictionary containing the average
                metrics for the model validation.
        """
        images = {
            'grid': [],
            'super_res': [],
            'low_res': [],
        }

        metrics = {
            'super_res': {'psnr': [], 'ssim': []},
            'inter': {'psnr': [], 'ssim': []}
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

                # metrics
                super_res_metrics = CalculateMetrics(reference, super_res)
                interpolated_metrics = CalculateMetrics(reference, interpolated)

                metrics['super_res']['psnr'].append(super_res_metrics.psnr())
                metrics['super_res']['ssim'].append(super_res_metrics.ssim())

                metrics['inter']['psnr'].append(interpolated_metrics.psnr())
                metrics['inter']['ssim'].append(interpolated_metrics.ssim())

                val_bar.set_description(
                    desc=(
                        f"[Converting LR images to SR images] "

                        f"Super PSNR: {super_res_metrics.psnr():.4f} dB "
                        f"Super SSIM: {super_res_metrics.ssim():.4f} "

                        f"Inter PSNR: {interpolated_metrics.psnr():.4f} dB "
                        f"Inter SSIM: {interpolated_metrics.ssim():.4f}"
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

            # return the average values for the metrics
            return {
                key: {subkey: statistics.mean(data)}
                for key, subdict in metrics.items()
                for subkey, data in subdict.items()
            }

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

        filepath = helpers.create_pth(f'{self.img_save_dir}/{subdirectory}')

        prepared_images = self._prepare_images(images, chunk_size)
        for index, image in enumerate(tqdm(prepared_images)):

            if grid:
                image = utils.make_grid(image, nrow=image_per_row)

            image_path = f'{filepath}/{str(index).zfill(4)}.png'
            utils.save_image(image, image_path)

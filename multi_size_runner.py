import logging

import pandas as pd
import torch
from torch.utils.data import DataLoader

import GAN

logger = logging.getLogger(__name__)


class MultiSizeRunner:
    """Checks multiple size combinations"""

    def __init__(self, combinations, param_path: str, blur_radius=0):
        self.combos = combinations
        self.params = param_path
        self.blur = blur_radius

    # fixme work on for loop decorator, pass in param path somehow
    def loop(self, fn):
        """
        Create
        """

        def wrapper(*args, **kwargs):
            for idx, (lr_size, hr_size) in enumerate(self.combos):
                self.params = f'{self.params}/{lr_size}-{hr_size}'
                fn(*args, **kwargs)
            return fn

        return wrapper

    # @loop
    def training(
            self,
            dataset_path: str,
            num_epochs: int = 5,
            csv_out: str = None
    ):

        # fixme unsure how to decouple result dict
        loss_dict = {
            # combine 2 lists for the 2 scores, must be equal length to total epochs
            'Metric': ['GL' for _ in range(num_epochs)] + ['DL' for _ in range(num_epochs)],
            'Epoch': [i + 1 for _ in range(2) for i in range(num_epochs)]
        }

        for idx, (lr_size, hr_size) in enumerate(self.combos):
            torch.cuda.empty_cache()

            # log sizes processed
            logger.info(
                f'[{idx + 1}/{len(self.combos)}] '
                f'Training: {lr_size}to{hr_size} '
            )

            # create generator and model objects
            # fixme have generator be created elsewhere?
            gen_net = GAN.Generator(hr_size // lr_size)
            model = GAN.RunModel(
                gen_net,
                f'{self.params}/{lr_size}to{hr_size}'
            )

            # training
            # fixme TrainDataset is the only essential that needs combos
            #   generate somewhere else to reduce coupling
            processed_data = GAN.TrainDataset(
                dataset_dir=dataset_path,
                high_res_size=hr_size,
                low_res_size=lr_size,
                blur=self.blur
            )
            loaded_data = DataLoader(
                dataset=processed_data,
                num_workers=4,
                batch_size=64,
                shuffle=True
            )
            loss_dict[f'{lr_size}-{hr_size}'] = model.run_training(
                dataset=loaded_data,
                num_epochs=num_epochs,
            )

        # save to dataframe and csv
        df = pd.DataFrame(loss_dict)
        logger.debug(df.head())

        if csv_out is None:
            return df

        df.to_csv(csv_out, index=False)

    def validation(self, dataset_path, img_save_dir, result_csv: str = None):
        results_dict = {'Metric': ['SR_PSNR', 'Inter_PSNR', 'SR_SSIM', 'Inter_SSIM']}

        for idx, (lr_size, hr_size) in enumerate(self.combos):
            torch.cuda.empty_cache()
            logger.info(f'validating: {lr_size} to {hr_size}')

            # print combination
            logger.info(
                f'[{idx + 1}/{len(self.combos)}] '
                f'Validation: {lr_size}-{hr_size} '
            )

            val_data = GAN.ValDataset(
                dataset_dir=dataset_path,
                high_res_size=hr_size,
                low_res_size=lr_size,
                blur=self.blur
            )
            val_loaded = DataLoader(
                dataset=val_data,
                num_workers=2,
                batch_size=1,
            )

            gen_param = f'{self.params}/' \
                        f'{lr_size}to{hr_size}/' \
                        f'gen_params.pth'

            # create generator; load params; set to eval
            gen_net = GAN.Generator(hr_size // lr_size)
            model = GAN.RunModel(gen_net, gen_param)

            SR_metrics, inter_metrics = model.run_validation(
                dataset=val_loaded,
                image_save_path=f'{img_save_dir}/{lr_size}to{hr_size}',
            )
            results_dict[f'{lr_size}-{hr_size}'] = [
                SR_metrics.psnr,
                inter_metrics.psnr,
                SR_metrics.ssim,
                inter_metrics.ssim
            ]

            logger.debug(results_dict.items())

        df = pd.DataFrame(results_dict)
        logger.debug(df.head())

        if result_csv is None:
            return df

        df.to_csv(result_csv)

import logging

import pandas as pd
import torch
from torch.utils.data import DataLoader

import gan

logger = logging.getLogger(__name__)


class MultiSizeRunner:
    """Checks multiple size combinations"""

    def __init__(
            self,
            combinations,
            param_path: str,
            gen_param_filename: str,
            disc_param_filename: str,
            blur_radius=0
    ):
        self.combos = combinations
        self.param_path = param_path
        self.gen_filename = gen_param_filename
        self.disc_filename = disc_param_filename
        self.blur = blur_radius

    def training(
            self,
            dataset_path: str,
            num_epochs: int = 5,
            csv_out: str = None
    ) -> pd.DataFrame:

        loss_dict = {
            # combine 2 lists for the 2 scores, must be equal length to total epochs
            'Metric': [
                          'GL' for _ in range(num_epochs)
                      ] + [
                          'DL' for _ in range(num_epochs)
                      ],
            'Epoch': [i + 1 for _ in range(2) for i in range(num_epochs)]
        }

        for idx, (lr_size, hr_size) in enumerate(self.combos):
            torch.cuda.empty_cache()

            # log sizes processed
            logger.info(
                f'[{idx + 1}/{len(self.combos)}] '
                f'Training: {lr_size}to{hr_size} '
            )
            # training
            processed_data = gan.TrainDataset(
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
            loss_dict[f'{lr_size}-{hr_size}'] = gan.Training(
                loaded_data,
                gan.Generator(hr_size // lr_size),
                gan.Discriminator(),
                f'{self.param_path}/{lr_size}to{hr_size}',
                total_epochs=num_epochs
            ).train_model()

        # save to dataframe and csv
        df = pd.DataFrame(loss_dict)
        logger.debug(df.head())

        df.to_csv(f'{csv_out}/{num_epochs}_epochs.csv', index=False)

        return df

    def validation(
            self,
            dataset_path,
            img_save_dir,
            result_csv: str = None
    ) -> pd.DataFrame:
        results_dict = {
            'Metric': ['SR_PSNR', 'Inter_PSNR', 'SR_SSIM', 'Inter_SSIM']
        }

        for idx, (lr_size, hr_size) in enumerate(self.combos):
            torch.cuda.empty_cache()
            logger.info(f'validating: {lr_size} to {hr_size}')

            # print combination
            logger.info(
                f'[{idx + 1}/{len(self.combos)}] '
                f'Validation: {lr_size}-{hr_size} '
            )

            val_data = gan.ValDataset(
                dataset_dir=dataset_path,
                high_res_size=hr_size,
                low_res_size=lr_size,
                blur=self.blur
            )
            loaded_data = DataLoader(
                dataset=val_data,
                num_workers=2,
                batch_size=1,
            )

            generator_param_path = f'{self.param_path}/' \
                                   f'{lr_size}to{hr_size}/' \
                                   f'{self.gen_filename}'

            model = gan.Validation(
                gan.Generator(hr_size // lr_size),
                generator_param_path,
                loaded_data,
                f'{img_save_dir}/{lr_size}to{hr_size}')

            model.main_loop()

            results_dict[f'{lr_size}-{hr_size}'] = model.collect_metrics

            logger.debug(results_dict.items())

        df = pd.DataFrame(results_dict)
        logger.debug(df.head())

        df.to_csv(result_csv)

        return df

import logging

import pandas as pd

import constants as c
import helpers
from .plate_reader import PlateReader


class Recognition:
    def __init__(self, plate_df: pd.DataFrame):
        self.df = plate_df

    def benchmark_recognition(self) -> None:
        """OCR recognition for the benchmark images."""
        for hr in [64, 128, 256, 512]:
            path = f'{c.BENCHMARK_PATH}/{hr}'
            self.df[f'{hr}_Benchmark'] = PlateReader(path).predictions()

        logging.info(f'Benchmark predictions finished')

    def low_resolution_recognition(self) -> None:
        """OCR recognition for the low resolution images."""
        for lr in c.LR_TO_TEST:
            path = f'{c.LOW_RES_PATH}/{lr}'
            lr_predictions = PlateReader(path).predictions()
            self.df[f'LR:{lr}'] = lr_predictions

        logging.info(f'LR predictions finished')

    def high_resolution_recognition(self) -> None:
        for lr, sr in helpers.combos(c.LR_TO_TEST, c.HR_TO_TEST):
            path = f'{c.SUPER_RES_PATH}/{lr}-{sr}'
            sr_predictions = PlateReader(path).predictions()
            self.df[f'SR:{lr}-{sr}'] = sr_predictions

        logging.info(f'SR predictions finished')

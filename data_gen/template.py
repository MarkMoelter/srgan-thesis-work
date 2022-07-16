from abc import ABC
from dataclasses import dataclass

import numpy as np


@dataclass
class Template(ABC):
    base_image_path: str

    x_pos: list
    y_pos: int

    available_chars: np.array

    font_size: int
    font_path: str


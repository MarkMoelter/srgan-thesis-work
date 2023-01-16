from abc import ABC
from dataclasses import dataclass
from typing import Type

from .label_properties import LabelProperties, PlateLabel


@dataclass
class PlateProperties(ABC):
    base_image_path: str
    available_chars: list[str]
    label_params: Type[LabelProperties]


@dataclass
class RectangularPlate(PlateProperties):
    base_image_path = 'data_gen/resources/plate_template.jpg'
    available_chars = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B',
        'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
        'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
    ]
    label_params = PlateLabel

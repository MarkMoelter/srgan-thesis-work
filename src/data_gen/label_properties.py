from abc import ABC
from dataclasses import dataclass

from .color import Color


@dataclass
class LabelProperties(ABC):
    x_pos: list[int]
    y_pos: int

    font_path: str
    font_size: int
    font_color: tuple[int, int, int]


@dataclass
class PlateLabel(LabelProperties):
    x_pos = [35, 87, 138, 190, 242, 294, 346]
    y_pos = 53

    font_path = 'data_gen/resources/plate_font.ttf'
    font_size = 120
    font_color = Color.BLACK

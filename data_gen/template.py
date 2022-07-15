from dataclasses import dataclass
import numpy as np


@dataclass
class Template:
    font_size = 120
    orig_plate_length = 434  # pixels
    orig_plate_height = 224  # pixels
    x_positions = [35, 87, 138, 190, 242, 294, 346]
    y_position = 53
    available_chars = np.array([
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B',
        'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
        'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
    ])
    font_path = 'data_gen/resources/plate_font.ttf'
    template_path = 'data_gen/resources/plate_template.jpg'

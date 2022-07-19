from dataclasses import dataclass

import numpy as np

from template import Template


@dataclass
class RectangularPlate(Template):
    base_image_path = '../resources/plate_template.jpg'

    x_pos = [35, 87, 138, 190, 242, 294, 346]
    y_pos = 53

    available_chars = np.array([
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B',
        'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
        'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
    ])

    font_size = 120
    font_path = '../resources/plate_font.ttf'

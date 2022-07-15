import random

from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm

from color import Color
from plate import Plate
from template import Template


class Generator:
    def __init__(self, template: Template):
        self.template = template

    def multi_plates(self, sets: int) -> list[Plate]:
        plate_list = []
        for _ in tqdm(range(sets)):
            for char in self.template.available_chars:
                for x in self.template.x_positions:
                    plate_list.append(self.single_plate(char, x))

        return plate_list

    def single_plate(self, char: str, x_pos: int) -> Plate:
        with Image.open(self.template.template_path).convert('RGBA') as base:
            label = self.generate_label(char, x_pos)
            label_str = ''.join(label)

            # create a transparent layer
            text_layer = Image.new(
                "RGBA",
                base.size,
                (*Color.BLACK, Color.TRANSPARENT)
            )

            # draw the label on the transparent layer
            for pos, val in zip(self.template.x_positions, label):
                ImageDraw.Draw(text_layer).text(
                    xy=(pos, self.template.y_position),
                    text=val,
                    font=ImageFont.truetype(
                        self.template.font_path,
                        self.template.font_size
                    ),
                    fill=(*Color.BLACK, Color.SOLID)
                )

            RGB_img = Image.alpha_composite(base, text_layer).convert('RGB')

            return Plate(label_str, RGB_img, label_str)

    def generate_label(self, character, x_pos) -> list[str]:
        label = []
        for x in self.template.x_positions:
            if x == x_pos:
                label.append(character)
            else:
                label.append(random.choice(self.template.available_chars))

        return label

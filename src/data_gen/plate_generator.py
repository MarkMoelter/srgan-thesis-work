import random

from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm
from typing import Type

from .color import Color
from .plate import Plate
from .plate_params import PlateParams


class PlateGenerator:
    def __init__(self, plate_template: Type[PlateParams]):
        self.template = plate_template

    def generate_plate_sets(self, sets: int = 1) -> list[Plate]:
        """
        Generate a list of plates for the given characters and positions available.

        :param sets: The number of set multiples to generate.
        :return: A list of plate objects.
        """

        return [
            self.generate_plate(self.generate_label(char, x))
            for x in self.template.label_params.x_pos
            for char in self.template.available_chars
            for _ in tqdm(range(sets))
        ]

    def generate_plate(self, label) -> Plate:
        """
        Generate a single Plate object given a character and a pixel position.

        :param label: The label to generate the plate with
        :return: Plate object with the given character in the given position.
        """
        with Image.open(self.template.base_image_path).convert('RGBA') as base:
            # create a transparent layer
            text_layer = Image.new(
                "RGBA",
                base.size,
                (*Color.BLACK, Color.TRANSPARENT)
            )

            # draw the label on the transparent layer
            for pos, val in zip(self.template.label_params.x_pos, label.split()):
                ImageDraw.Draw(text_layer).text(
                    xy=(pos, self.template.label_params.y_pos),
                    text=val,
                    font=ImageFont.truetype(
                        self.template.label_params.font_path,
                        self.template.label_params.font_size
                    ),
                    fill=(*Color.BLACK, Color.SOLID)
                )
            # combine the base and label layer together
            RGB_img = Image.alpha_composite(base, text_layer).convert('RGB')

            return Plate(label, RGB_img, label)

    def generate_label(self, character: str, x_position: int) -> str:
        """
        Generate a license plate label with the given character
        in the given position. All other characters will be randomly chosen
        from the list of characters in the plate template.

        Length of the label is equal to the number of x-positions
        in the plate template.

        :param character: Character that will go in the given position
        :param x_position: Position in pixel coordinates.
        :return: String label with the given character in the given position.
        """
        # invalid position
        if x_position not in self.template.label_params.x_pos:
            raise ValueError('Invalid position')
        # invalid character
        if character not in self.template.available_chars:
            raise ValueError('Invalid character')

        label = []
        for x in self.template.label_params.x_pos:
            if x == x_position:
                label.append(character)
            elif x_position in self.template.label_params.x_pos:
                label.append(random.choice(self.template.available_chars))

        return ''.join(label)

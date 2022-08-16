import random

from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm

from .color import Color
from .plate import Plate
from .plate_templates import Template


class PlateGenerator:
    def __init__(self, plate_template: Template):
        self.template = plate_template

    def multi_plates(self, sets: int = 1) -> list[Plate]:
        """
        Generate a list of plate objects.

        EQ: sets * characters * x-positions

        :param sets: The number of sets to generate.
        :return: A list of plate objects.
        """

        return [
            self.single_plate(char, x)
            for x in self.template.x_pos
            for char in self.template.available_chars
            for _ in tqdm(range(sets))
        ]

    def single_plate(self, character: str, x_position: int) -> Plate:
        """
        Generate a single Plate object given a character and a pixel position.

        :param character: Character to map to the x_pos param.
        :param x_position: Position in pixel coordinates.
        :return: Plate object with the given character in the given position.
        """
        self.char_error(character)
        self.pos_error(x_position)

        with Image.open(self.template.base_image_path).convert('RGBA') as base:
            label = self.generate_label(character, x_position)

            # create a transparent layer
            text_layer = Image.new(
                "RGBA",
                base.size,
                (*Color.BLACK, Color.TRANSPARENT)
            )

            # draw the label on the transparent layer
            for pos, val in zip(self.template.x_pos, label.split()):
                ImageDraw.Draw(text_layer).text(
                    xy=(pos, self.template.y_pos),
                    text=val,
                    font=ImageFont.truetype(
                        self.template.font_path,
                        self.template.font_size
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

        # raise Exceptions
        self.pos_error(x_position)
        self.char_error(character)

        label = []
        for x in self.template.x_pos:
            if x == x_position:
                label.append(character)
            elif x_position in self.template.x_pos:
                label.append(random.choice(self.template.available_chars))

        return ''.join(label)

    def char_error(self, character: str):
        if character not in self.template.available_chars:
            raise ValueError('Invalid character')

    def pos_error(self, position: int):
        if position not in self.template.x_pos:
            raise ValueError('Invalid position')

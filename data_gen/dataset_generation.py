import os
from dataclasses import dataclass
from random import randint

import numpy as np
import pandas as pd
from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm

import data_gen.funcs as f


class Color:
    BLACK = (0, 0, 0)
    TRANSPARENT = 0
    SOLID = 255

@dataclass
class Plate:
    font_size = 120
    x_position_list = [35, 87, 138, 190, 242, 294, 346]
    y_position = 53
    character_list = np.array([
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B',
        'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
        'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
    ])
    font_path = 'data_gen/plate_font.ttf'
    template_path = 'data_gen/plate_template.jpg'

def plate_gen(
        multiples,
        img_save_dir: str,
        label_csv_path: str = None
):
    """
    Generate a set of license plates.
    Each plate has 6 random characters and 1 predetermined character.
    Function saves plates as RGB.

    :param multiples: Multiples of 252 base license plates generated.
    :param img_save_dir: Directory for saved images.
    :param label_csv_path: Filename and directory for generated image filenames and labels.
    :return: DataFrame containing generated filenames and labels.
    """

    plate_list, idx = [], 0
    for _ in tqdm(range(multiples), desc=f'Multiples of {len(Plate.character_list) * len(Plate.x_position_list)}'):
        for char in Plate.character_list:
            for x_val in Plate.x_position_list:
                with Image.open(Plate.template_path).convert("RGBA") as base:

                    # transparent layer for text
                    text_layer = Image.new(
                        "RGBA",
                        base.size,
                        (*Color.BLACK, Color.TRANSPARENT)
                    )

                    # generate character list and draw on text layer
                    plate_label = []
                    for x in Plate.x_position_list:
                        # label list generation
                        if x_val == x:
                            plate_label.append(char)
                        else:
                            plate_label.append(Plate.character_list[randint(0, 35)])
                        # draw characters
                        ImageDraw.Draw(text_layer).text(
                            xy=(x, Plate.y_position),
                            text=plate_label[-1],
                            font=ImageFont.truetype(
                                Plate.font_path,
                                Plate.font_size
                            ),
                            fill=(*Color.BLACK, Color.SOLID)
                        )

                    # combine the 2 layers together
                    RGB_img = Image.alpha_composite(base, text_layer).convert('RGB')
                    RGB_img = RGB_img.resize((512, 512))  # resize to be (512, 512)

                    # filenames
                    filename = str(idx).zfill(4)
                    idx += 1

                    f.create_pth(img_save_dir)  # create dir if not exist
                    RGB_img.save(f'{img_save_dir}/{filename}.png')  # save image

                    combined_list = [filename, ''.join(plate_label)]
                    plate_list.append(combined_list)

    # save plate list as csv
    col_names = ['filename', 'Plate Labels']
    df = pd.DataFrame(plate_list, columns=col_names)

    if label_csv_path:
        df.to_csv(label_csv_path, index=False)
    return df


if __name__ == "__main__":
    os.chdir(r'C:\Users\marke\PycharmProjects\Thesis SRGAN')

    plate_dir = 'Results/dataset/testing_set'
    master_csv = f'Results/CSVs/Master_Dataset.csv'

    sets = 6

    # generate plates
    df = plate_gen(sets, img_save_dir=plate_dir)

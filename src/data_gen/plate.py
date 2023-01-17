from PIL import Image

import src.utils.path as path


class Plate:
    def __init__(self, filename: str, image: Image, label: str):
        self.filename = filename
        self.image = image
        self.label = label

    def __repr__(self):
        return f'Plate({self.filename}, {self.label})'

    def save_plate(self, directory: str) -> None:
        """
        Save the image to the given directory.

        :param directory: Folder path to save the image to.
        """
        path.create_path(directory)
        self.image.save(f'{directory}/{self.filename}.png')

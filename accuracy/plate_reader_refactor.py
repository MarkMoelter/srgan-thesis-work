import logging
import os
from tqdm import tqdm
import cv2

import pandas as pd
import pytesseract as tess
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)

class Reader:
    def __init__(self):
        pass

    def filter_string(self):
        pass

    def transform(self):
        """Implements the transforms on an image"""
        pass

class Plate:
    def __init__(self, filename, label, prediction):
        self.filename = filename
        self.label = label
        self.prediction = prediction

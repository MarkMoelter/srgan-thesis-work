import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


def reshape_tensor(tensor: np.ndarray, order=(2, 3, 1)) -> np.ndarray:
    """
    Reshapes tensor into grayscale image array.

    :param tensor: Tensor to be transformed
    :param order: order of tensors
    :return: Grayscale array
    """
    reshape_list = [tensor.shape[shape] for shape in order]
    return tensor.reshape(reshape_list)


def tensor_to_grayscale(tensor: np.ndarray) -> np.ndarray:
    """
    Flattens an RGB tensor into a 2-dimensional grayscale array.

    :param tensor: RGB tensor to convert.
    :return: A 2-dimensional grayscale image.
    """
    array = tensor.cpu().squeeze(0).numpy()
    gray_array = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    return gray_array


class CalculateMetrics:
    """
    Calculate image quality metrics
    """

    def __init__(self, reference: np.ndarray, prediction: np.ndarray):
        self.reference = reference
        self.prediction = prediction

    def psnr(self) -> float:
        """Implements PSNR from skimage.metrics"""
        val = peak_signal_noise_ratio(
            reshape_tensor(self.reference),
            reshape_tensor(self.prediction)
        )
        return val

    def ssim(self) -> float:
        """Implements SSIM from skimage.metrics"""
        val, _ = structural_similarity(
            reshape_tensor(self.reference),
            reshape_tensor(self.prediction),
            full=True
        )
        return val

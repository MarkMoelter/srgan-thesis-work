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


def tensor_to_grayscale(tensor: np.array) -> np.ndarray:
    """
    Flattens an RGB tensor into a 2-dimensional grayscale array.

    :param tensor: RGB tensor to convert.
    :return: A 2-dimensional grayscale image.
    """
    # put the array on the cpu,
    array = tensor.cpu().squeeze(0).numpy()
    return cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)

def ssim(reference, prediction) -> float:
    """Implements SSIM from skimage.metrics"""
    val, _ = structural_similarity(
        reshape_tensor(reference),
        reshape_tensor(prediction),
        full=True
    )
    return val

def psnr(reference, prediction) -> float:
    """Implements PSNR from skimage.metrics"""
    val = peak_signal_noise_ratio(
        reshape_tensor(reference),
        reshape_tensor(prediction)
    )
    return val

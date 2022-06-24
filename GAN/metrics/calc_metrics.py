import cv2
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


def tensor_preprocess(tensor, order=(2, 3, 1)):
    """
    Reshapes tensor into extractable image array
    :param tensor: Tensor to be transformed
    :param order: order of tensors
    :return: Grayscale array
    """
    reshape_list = [tensor.shape[shape] for shape in order]
    reshaped_tensor = tensor.reshape(reshape_list)
    array = reshaped_tensor.cpu().squeeze(0).numpy()
    return cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)

class CalcMetrics:
    """
    Calculate image quality metrics
    """
    def __init__(self, reference, prediction):
        self.reference = reference
        self.prediction = prediction


    def psnr(self) -> float:
        """Implements PSNR from skimage.metrics"""
        val = peak_signal_noise_ratio(
            tensor_preprocess(self.reference),
            tensor_preprocess(self.prediction)
        )
        return val

    def ssim(self) -> float:
        """Implements SSIM from skimage.metrics"""
        val, _ = structural_similarity(
            tensor_preprocess(self.reference),
            tensor_preprocess(self.prediction),
            full=True
        )
        return val
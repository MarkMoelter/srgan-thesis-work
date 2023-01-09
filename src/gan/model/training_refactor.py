import torch


class Train:
    """Train the model"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self):
        pass


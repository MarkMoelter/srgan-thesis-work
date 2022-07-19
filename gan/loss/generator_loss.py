import torch
from torchvision.models.vgg import vgg16

from .total_variance_loss import TotalVarianceLoss


class GeneratorLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(pretrained=True)
        loss_network = torch.nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = torch.nn.MSELoss()
        self.tv_loss = TotalVarianceLoss()

    def forward(self, out_labels, out_images, target_images):
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)

        # Adversarial Loss
        adversarial_loss = 0.001 * torch.mean(1 - out_labels)

        # Perception Loss
        perception_loss = 0.006 * self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))

        # TV Loss
        tv_loss = 2e-8 * self.tv_loss(out_images)

        return image_loss + adversarial_loss + perception_loss + tv_loss
#     MSE + w1*Adversarial + w2*MSE(vgg16 pretrained) + w3*totalVar

import logging
import statistics as stats

import torch
from tqdm import tqdm

import helpers
from gan.loss import GeneratorLoss

logger = logging.getLogger(__name__)


class Training:
    """Implement each step in the model training phase"""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self, gen_net, disc_net, param_dir):
        self.gen_net = gen_net.to(self.device)
        self.disc_net = disc_net.to(self.device)

        self.param_dir = param_dir

    def train_model(self, dataset, total_epochs) -> tuple[list[float], list[float]]:
        # optimizers
        gen_loss_each_epoch = []
        disc_loss_each_epoch = []
        for epoch in range(1, 1 + total_epochs):
            self.gen_net.train()
            self.disc_net.train()

            train_bar = tqdm(dataset)
            avg_disc_loss, avg_gen_loss = self._train_loop(train_bar)

            # set bar description
            train_bar.set_description(
                desc=(
                    f"[{epoch}/{total_epochs}] "
                    f"Loss_D: {avg_gen_loss:.4f} "
                    f"Loss_G: {avg_disc_loss:.4f} "
                )
            )

            logger.info(f'[{epoch}/{total_epochs}] Gen Loss: {avg_gen_loss}')
            logger.info(f'[{epoch}/{total_epochs}] Disc Loss: {avg_disc_loss}')
            gen_loss_each_epoch.append(avg_gen_loss)
            disc_loss_each_epoch.append(avg_disc_loss)

        # save params
        helpers.create_pth(self.param_dir)
        torch.save(self.gen_net.state_dict(), f'{self.param_dir}/gen_params.pth')
        torch.save(self.disc_net.state_dict(), f'{self.param_dir}/disc_params.pth')

        return disc_loss_each_epoch, gen_loss_each_epoch

    def _train_loop(self, dataset) -> tuple[float, float]:
        """
        The loop that trains the model. Calculates the loss of the model after each iteration.
        Returns the model's average loss over the dataset.

        :param dataset: The dataset to process.
        :return: The average value for the discriminator and generator loss in that order.
        """
        gen_optim = torch.optim.Adam(self.gen_net.parameters(), lr=0.002)
        disc_optim = torch.optim.Adam(self.disc_net.parameters(), lr=0.002)

        gen_loss_values = []
        disc_loss_values = []
        for gen_input, target_img in dataset:
            # put image tensors on the device
            out_img = self.gen_net(
                torch.autograd.Variable(gen_input).to(self.device)
            )
            target_img = torch.autograd.Variable(target_img).to(self.device)

            # compute loss
            out_label = self._get_label(out_img)
            target_label = self._get_label(target_img)

            disc_loss = self._update_discriminator(out_label, target_label)
            gen_loss = self._update_generator(out_label, out_img, target_img)

            # update optimizers
            gen_optim.step()
            disc_optim.step()

            disc_loss_values.append(disc_loss)
            gen_loss_values.append(gen_loss)

        return stats.mean(disc_loss_values), stats.mean(gen_loss_values)

    def _get_label(self, img: torch.Tensor) -> torch.Tensor:
        """Get the label of an image from the discriminator."""
        self.disc_net.zero_grad()
        return self.disc_net(img).mean()

    def _update_generator(self, out_label, out_img, target_img):
        """Update the generator after calculating the loss."""
        self.gen_net.zero_grad()
        # compute loss
        loss_fn = GeneratorLoss().to(self.device)
        g_loss = loss_fn(out_label, out_img, target_img)
        # backprop
        g_loss.backward()

        return g_loss.item()

    @staticmethod
    def _update_discriminator(out_label, target_label):
        """Update the discriminator after calculating the loss."""
        # compute loss
        d_loss = 1 - target_label + out_label
        # backprop
        d_loss.backward(retain_graph=True)
        return d_loss.item()

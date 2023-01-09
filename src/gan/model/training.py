import logging
import statistics as stats

import torch
from tqdm import tqdm

import helpers
from gan.loss import GeneratorLoss

logger = logging.getLogger(__name__)


# todo: refactor into useful methods instead of breaking up a large function into different parts of a huge class.
class Training:
    """Implement each step in the model training phase"""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self, dataset, gen_net, disc_net, param_dir, epoch_count):
        self.dataset = dataset
        self.gen_net = gen_net.to(self.device)
        self.disc_net = disc_net.to(self.device)
        self.gen_optimizer = torch.optim.Adam(self.gen_net.parameters(), lr=0.002)
        self.disc_optimizer = torch.optim.Adam(self.disc_net.parameters(), lr=0.002)

        self.param_dir = param_dir
        self.epoch_count = epoch_count

    def train_model(self) -> tuple[list[float], list[float]]:
        # optimizers
        gen_loss = []
        disc_loss = []
        for epoch in range(1, 1 + self.epoch_count):
            d_loss, g_loss = self._train_epoch(epoch)

            gen_loss.append(g_loss)
            disc_loss.append(d_loss)

        # save params
        helpers.create_pth(self.param_dir)
        torch.save(self.gen_net.state_dict(), f'{self.param_dir}/gen_params.pth')
        torch.save(self.disc_net.state_dict(), f'{self.param_dir}/disc_params.pth')

        return disc_loss, gen_loss

    def _train_epoch(self, epoch):
        # todo: remove epoch parameter

        self.gen_net.train()
        self.disc_net.train()

        train_bar = tqdm(self.dataset)
        gen_loss_values, disc_loss_values = self._train_loop(train_bar)

        avg_gen_loss = stats.mean(gen_loss_values)
        avg_disc_loss = stats.mean(disc_loss_values)

        # set bar description
        train_bar.set_description(
            desc=(
                f"[{epoch}/{self.epoch_count}] "
                f"Loss_D: {avg_gen_loss:.4f} "
                f"Loss_G: {avg_disc_loss:.4f} "
            )
        )
        # todo: another bad use of epoch
        logger.info(f'[{epoch}/{self.epoch_count}] Gen Loss: {avg_gen_loss}')
        logger.info(f'[{epoch}/{self.epoch_count}] Disc Loss: {avg_disc_loss}')

        return avg_disc_loss, avg_gen_loss

    def _train_loop(self, dataset) -> tuple[list[float], list[float]]:
        """The loop that trains the model"""
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
            self.gen_optimizer.step()
            self.disc_optimizer.step()

            disc_loss_values.append(disc_loss)
            gen_loss_values.append(gen_loss)

            return gen_loss_values, disc_loss_values

    def _get_label(self, img: torch.Tensor) -> torch.Tensor:
        """Get the label of an image from the discriminator."""
        self.disc_net.zero_grad()
        return self.disc_net(img).mean()

    @staticmethod
    def _update_discriminator(out_label, target_label):
        """Update the discriminator after calculating the loss."""
        # compute loss
        d_loss = 1 - target_label + out_label
        # backprop
        d_loss.backward(retain_graph=True)
        return d_loss.item()

    def _update_generator(self, out_label, out_img, target_img):
        """Update the generator after calculating the loss."""
        self.gen_net.zero_grad()
        # compute loss
        loss_fn = GeneratorLoss().to(self.device)
        g_loss = loss_fn(out_label, out_img, target_img)
        # backprop
        g_loss.backward()

        return g_loss.item()

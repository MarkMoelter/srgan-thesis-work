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

        gen_loss = []
        disc_loss = []

        train_bar = tqdm(self.dataset)
        for gen_input, real_img in train_bar:
            # collect images
            z = torch.autograd.Variable(gen_input).to(self.device)
            fake_img = self.gen_net(z)

            real_img = torch.autograd.Variable(real_img).to(self.device)

            # compute loss
            d_loss, g_loss = self._update_nets(real_img, fake_img)

            # update optimizers
            self.gen_optimizer.step()
            self.disc_optimizer.step()

            # todo: move all messages into a separate class? bad use of epoch
            # set bar description
            train_bar.set_description(
                desc=(
                    f"[{epoch}/{self.epoch_count}] "
                    f"Loss_D: {g_loss:.4f} "
                    f"Loss_G: {d_loss:.4f} "
                )
            )

            disc_loss.append(d_loss)
            gen_loss.append(g_loss)

        avg_gen_loss = stats.mean(gen_loss)
        avg_disc_loss = stats.mean(disc_loss)

        # todo: another bad use of epoch
        logger.info(f'[{epoch}/{self.epoch_count}] Gen Loss: {avg_gen_loss}')
        logger.info(f'[{epoch}/{self.epoch_count}] Disc Loss: {avg_disc_loss}')

        return avg_disc_loss, avg_gen_loss

    def _update_nets(self, real_img, fake_img):
        """Update generator and discriminator networks"""

        ### update discriminator ###
        self.disc_net.zero_grad()
        # collect real and fake classifications
        real_label = self.disc_net(real_img).mean()
        fake_label = self.disc_net(fake_img).mean()
        # compute loss
        d_loss = 1 - real_label + fake_label
        # backprop
        d_loss.backward(retain_graph=True)

        ### update generator ###
        self.gen_net.zero_grad()
        # compute loss
        loss_fn = GeneratorLoss().to(self.device)
        g_loss = loss_fn(fake_label, fake_img, real_img)
        # backprop
        g_loss.backward()

        # return loss tuple
        return d_loss.item(), g_loss.item()

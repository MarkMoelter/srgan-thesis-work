import logging
import statistics as stats

import torch
from tqdm import tqdm

import helpers
from gan.loss import GeneratorLoss
from gan.metrics import TrainMetrics

logger = logging.getLogger(__name__)


# FIXME Refactor this abomination to work as separate methods.
class Training:
    """Implement each step in the model training phase"""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self, dataset, gen_net, disc_net, param_dir, epoch_count):
        self.dataset = dataset
        self.gen_net = gen_net.to(self.device)
        self.disc_net = disc_net.to(self.device)
        self.param_dir = param_dir
        self.epoch_count = epoch_count

    def main_loop(self):
        # optimizers
        gen_optimizer = torch.optim.Adam(self.gen_net.parameters(), lr=0.002)
        disc_optimizer = torch.optim.Adam(self.disc_net.parameters(), lr=0.002)

        data_collect = TrainMetrics()
        for epoch in range(1, 1 + self.epoch_count):
            d_loss, g_loss = self.train_epoch(epoch, gen_optimizer, disc_optimizer)

            data_collect.gen_loss.append(g_loss)
            data_collect.disc_loss.append(d_loss)

        # save params
        helpers.create_pth(self.param_dir)
        torch.save(self.gen_net.state_dict(), f'{self.param_dir}/gen_params.pth')
        torch.save(self.disc_net.state_dict(), f'{self.param_dir}/disc_params.pth')

        return data_collect.disc_loss, data_collect.gen_loss

    def train_epoch(self, epoch, gen_optim, disc_optim):
        self.gen_net.train()
        self.disc_net.train()

        metric_collect = TrainMetrics()

        train_bar = tqdm(self.dataset)
        for gen_input, real_img in train_bar:
            # collect images
            z = torch.autograd.Variable(gen_input).to(self.device)
            fake_img = self.gen_net(z)

            real_img = torch.autograd.Variable(real_img).to(self.device)

            # compute loss
            d_loss, g_loss = self.update_nets(real_img, fake_img)

            # update optimizers
            gen_optim.step()
            disc_optim.step()

            # set bar description
            train_bar.set_description(
                desc=(
                    f"[{epoch}/{self.epoch_count}] "
                    f"Loss_D: {g_loss:.4f} "
                    f"Loss_G: {d_loss:.4f} "
                )
            )

            metric_collect.disc_loss.append(d_loss)
            metric_collect.gen_loss.append(g_loss)

        metric_collect.gen_loss = stats.mean(metric_collect.gen_loss)
        metric_collect.disc_loss = stats.mean(metric_collect.disc_loss)
        logger.info(f'[{epoch}/{self.epoch_count}] Gen Loss: {metric_collect.gen_loss}')
        logger.info(f'[{epoch}/{self.epoch_count}] Disc Loss: {metric_collect.disc_loss}')

        return metric_collect.disc_loss, metric_collect.gen_loss

    def update_nets(self, real_img, fake_img):
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

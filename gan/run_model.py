import logging
import statistics as stats

import torch
from tqdm import tqdm

import gan.loss as loss
import gan.metrics as metrics
import gan.model as model
import helpers

logger = logging.getLogger(__name__)


class RunModel:
    """Handles training and validation of the model."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self, gen, param_path):
        self.gen = gen.to(self.device)
        self.param_path = param_path

    def run_training(self, dataset, num_epochs):
        loss_gen = loss.GeneratorLoss().to(self.device)
        disc_net = model.Discriminator().to(self.device)

        # optim init
        gen_optimizer = torch.optim.Adam(self.gen.parameters(), lr=0.002)
        disc_optimizer = torch.optim.Adam(disc_net.parameters(), lr=0.002)

        # create a metric object
        outer_metrics = metrics.TrainMetrics()

        for epoch in range(1, num_epochs + 1):
            self.gen.train()
            disc_net.train()
            inner_metrics = metrics.TrainMetrics()

            train_bar = tqdm(dataset)
            for gen_input, real_img in train_bar:
                # generate fake image
                z = torch.autograd.Variable(gen_input).cuda()
                fake_img = self.gen(z)

                # real image
                real_img = torch.autograd.Variable(real_img).cuda()

                #######################
                # (1) Update D network: maximize D(x)-1-D(G(z))
                #######################
                disc_net.zero_grad()

                real_label = disc_net(real_img).mean()
                fake_label = disc_net(fake_img).mean()

                d_loss = 1 - real_label + fake_label
                d_loss.backward(retain_graph=True)

                #######################
                # (2) Update G network: minimize 1-D(G(z))
                # + Perception Loss + Image Loss + TV Loss
                #######################
                self.gen.zero_grad()

                g_loss = loss_gen(fake_label, fake_img, real_img)
                g_loss.backward()

                gen_optimizer.step()
                disc_optimizer.step()

                # collect results
                inner_metrics.gen_loss.append(g_loss.item())
                inner_metrics.disc_loss.append(d_loss.item())

                # set description
                train_bar.set_description(
                    desc=(
                        f"[{epoch}/{num_epochs}] "
                        f"Loss_D: {inner_metrics.disc_loss[-1]:.4f} "
                        f"Loss_G: {inner_metrics.gen_loss[-1]:.4f} "
                    )
                )

            outer_metrics.gen_loss.append(stats.mean(inner_metrics.gen_loss))
            outer_metrics.disc_loss.append(stats.mean(inner_metrics.disc_loss))

            logger.info(
                f'[{epoch}/{num_epochs}] Gen Loss: {outer_metrics.gen_loss}'
            )
            logger.info(
                f'[{epoch}/{num_epochs}] Disc Loss: {outer_metrics.disc_loss}'
            )

        # save params
        helpers.create_pth(self.param_path)
        torch.save(self.gen.state_dict(), f'{self.param_path}/gen_params.pth')
        torch.save(disc_net.state_dict(), f'{self.param_path}/disc_params.pth')

        logger.info('Model parameters saved')

        return outer_metrics.gen_loss + outer_metrics.disc_loss

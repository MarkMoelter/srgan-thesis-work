import logging
import statistics as stats

import torch
from tqdm import tqdm

import loss
import model
import metrics
import helpers

logger = logging.getLogger(__name__)


class Images:
    def __init__(self):
        self.grid = []
        self.low_res = []
        self.super_res = []


# todo replace with model training and validation
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
            logger.info(f'[{epoch}/{num_epochs}] Gen Loss: {outer_metrics.gen_loss}')
            logger.info(f'[{epoch}/{num_epochs}] Disc Loss: {outer_metrics.disc_loss}')

        # save params
        helpers.create_pth(self.param_path)
        torch.save(self.gen.state_dict(), f'{self.param_path}/gen_params.pth')
        torch.save(disc_net.state_dict(), f'{self.param_path}/disc_params.pth')

        logger.info('Model parameters saved')

        return outer_metrics.gen_loss + outer_metrics.disc_loss

    def run_validation(self, dataset, image_save_path):
        img_cls = Images()

        # store PSNR and SSIM in objects
        outer_SR_metrics = metrics.ValMetrics()
        outer_inter_metrics = metrics.ValMetrics()

        # loading generator and params
        self.gen.load_state_dict(torch.load(self.param_path))
        self.gen.eval()

        val_bar = tqdm(dataset)
        with torch.no_grad():
            for idx, (low_res, interpolated, target) in enumerate(val_bar):
                # send tensors to cuda
                target = target.to(self.device)
                super_res = self.gen(low_res.to(self.device))
                interpolated = interpolated.to(self.device)

                ### Calculate and Store Metrics ###
                # Interpolated Image Results
                inter_metrics = metrics.CalcMetrics(target, interpolated)
                inter_psnr = inter_metrics.psnr()
                inter_ssim = inter_metrics.ssim()

                # Super Res Image Results
                inner_SR_metrics = metrics.CalcMetrics(target, super_res)
                SR_psnr = inner_SR_metrics.psnr()
                SR_ssim = inner_SR_metrics.ssim()

                # update val bar desc
                val_bar.set_description(
                    desc=(
                        f"[converting LR images to SR images] "
                        f"Super PSNR: {SR_psnr:.4f} dB "
                        f"Inter PSNR: {inter_psnr:.4f} dB "
                        f"Super SSIM: {SR_ssim:.4f} "
                        f"Inter SSIM: {inter_ssim:.4f}"
                    )
                )

                # append metrics to outer objects
                outer_SR_metrics.psnr.append(SR_psnr)
                outer_SR_metrics.ssim.append(SR_ssim)

                outer_inter_metrics.psnr.append(inter_psnr)
                outer_inter_metrics.ssim.append(inter_ssim)

                ### Save Images ###
                img_cls.grid.extend(
                    [
                        target.data.cpu().squeeze(0),
                        interpolated.cpu().squeeze(0),
                        super_res.data.cpu().squeeze(0)
                    ]
                )
                img_cls.super_res.extend([super_res.data.cpu().squeeze(0)])
                img_cls.low_res.extend([low_res.data.cpu().squeeze(0)])

            # take the mean for the results
            outer_inter_metrics.psnr = stats.mean(outer_inter_metrics.psnr)
            outer_inter_metrics.ssim = stats.mean(outer_inter_metrics.ssim)

            outer_SR_metrics.psnr = stats.mean(outer_SR_metrics.psnr)
            outer_SR_metrics.ssim = stats.mean(outer_SR_metrics.ssim)

            # fixme collect all image saves into a single def or class
            # grid
            img_cls.grid = torch.stack(img_cls.grid)
            img_cls.grid = torch.chunk(
                img_cls.grid,
                len(val_bar)
            )
            helpers.img_save(
                img_cls.grid,
                f'{image_save_path}/grid',
                desc='[saving grid images]',
                grid=True,
                nrow=3
            )

            # SR
            img_cls.super_res = torch.stack(img_cls.super_res)
            img_cls.super_res = torch.chunk(
                img_cls.super_res,
                len(val_bar)
            )
            helpers.img_save(
                img_cls.super_res,
                f'{image_save_path}/SR',
                desc='[saving SR images]')

            # LR
            img_cls.low_res = torch.stack(img_cls.low_res)
            img_cls.low_res = torch.chunk(
                img_cls.low_res,
                len(val_bar)
            )
            helpers.img_save(
                img_cls.low_res,
                f'{image_save_path}/LR',
                desc='[saving LR images]')

            return outer_SR_metrics, outer_inter_metrics

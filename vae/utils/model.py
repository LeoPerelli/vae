from torch import nn
import torch
import torch.nn.functional as F
from vae.utils.samplers import MultivariateNormal


class ResidualEncoderBlock(nn.Module):
    def __init__(
        self, in_channels, kernel_size=3, is_first_block=False, stride=1, residual=True, out_channels=None
    ):

        super().__init__()
        self.is_first_block = is_first_block
        self.residual = residual

        stride = 1
        padding = "same"

        out_channels = in_channels if out_channels is None else out_channels
        if self.is_first_block:
            stride = 2
            padding = 1
            out_channels = 2 * in_channels
            self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding="same"
        )

    def forward(self, x):

        out1 = F.relu(self.conv1(x))
        out2 = F.relu(self.conv2(out1))

        if self.is_first_block and self.residual:
            x = self.avg_pool(x)  # B x C x H x W -> # B x C x H/2 x W/2
            x = torch.cat((x, torch.zeros_like(x)), dim=1)  # B x C x H/2 x W/2 -> # B x 2C x H/2 x W/2

        if self.residual:
            return x + out2
        else:
            return out2


class ResidualEncoder(nn.Module):
    def __init__(self, in_channels_start, depth):

        super().__init__()
        self.encoder_blocks = nn.ModuleList([])
        channels = in_channels_start

        self.encoder_blocks.append(ResidualEncoderBlock(in_channels=3, out_channels=channels, residual=False))

        for _ in range(depth):
            self.encoder_blocks.extend(
                (
                    ResidualEncoderBlock(channels, is_first_block=True),
                    ResidualEncoderBlock(2 * channels),
                    ResidualEncoderBlock(2 * channels),
                )
            )
            channels = int(channels * 2)

        self.encoder_blocks.append(nn.Conv2d(channels, 2, 1))
        self.encoder_blocks.append(
            nn.Flatten(start_dim=2)
        )  # flatten the feature maps to B x C (=2, which are log mu and log ) x latent_dim

    def forward(self, x):

        for block in self.encoder_blocks:
            x = block(x)

        log_mu = x[:, [0], :]
        log_sigma = x[:, [1], :]

        return log_mu, log_sigma


class ResidualDecoderBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=3, is_last_block=False, out_channels=None):

        super().__init__()
        self.is_last_block = is_last_block

        padding = 1
        stride = 1
        output_padding = 0

        if self.is_last_block:
            stride = 2
            output_padding = 1
            out_channels = int(in_channels / 2) if out_channels is None else out_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.residual = not is_last_block and not out_channels != in_channels

        self.conv1 = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=1,
            stride=1,
        )
        self.conv2 = nn.ConvTranspose2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )

    def forward(self, x):

        out1 = F.relu(self.conv1(x))
        out2 = F.relu(self.conv2(out1))

        if self.residual:
            return x + out2
        else:
            return out2


class ResidualDecoder(nn.Module):
    def __init__(self, in_channels_start, depth):

        super().__init__()
        self.decoder_blocks = nn.ModuleList([])
        channels = in_channels_start

        self.decoder_blocks.append(
            ResidualDecoderBlock(in_channels=1, out_channels=channels, is_last_block=False)
        )

        for _ in range(depth):
            self.decoder_blocks.extend(
                (
                    ResidualDecoderBlock(channels),
                    ResidualDecoderBlock(channels),
                    ResidualDecoderBlock(channels, is_last_block=True),
                )
            )
            channels = int(channels / 2)

        self.decoder_blocks.append(
            nn.Conv2d(channels, 6, 1)
        )  # 6 output channels: one per RGB channel, and for each i have log_mu and log_sigma
        self.decoder_blocks.append(nn.Flatten(start_dim=2))  # flatten the feature maps to B x 6 x latent_dim

    def forward(self, x):

        for block in self.decoder_blocks:
            x = block(x)

        log_mu = x[:, :3, :]
        log_sigma = x[:, 3:, :]

        return log_mu, log_sigma


class VariationalAutoEncoder(nn.Module):
    def __init__(self, encoder_decoder_depth, encoder_start_channels, img_dim = 256):

        super().__init__()

        self.encoder = ResidualEncoder(in_channels_start=encoder_start_channels, depth=encoder_decoder_depth)
        self.decoder = ResidualDecoder(
            in_channels_start=int(encoder_start_channels * (2) ** encoder_decoder_depth),
            depth=encoder_decoder_depth,
        )
        self.latent_dim = int(img_dim * 0.5** encoder_decoder_depth) ** 2
        self.pixel_dim = img_dim * img_dim
        self.encoder_sampler = MultivariateNormal(dimension=self.latent_dim)
        self.decoder_sampler = MultivariateNormal(dimension=self.pixel_dim)

    def generate(self):

        z = torch.unflatten(
            input=self.encoder_sampler.sample_isotropic()[None],
            dim=1,
            sizes=(int(self.latent_dim**0.5), -1),
        )[:, None, :, :].to(next(self.parameters()).device)
        log_mu, log_sigma = self.decoder(z)
        flattened_x = self.decoder_sampler.sample(log_mu, log_sigma).to(next(self.parameters()).device)
        x = torch.unflatten(flattened_x, dim=2, sizes=(int(self.pixel_dim**0.5), -1))

        return x

    def compute_loss(self, x, n_samples=1):

        log_mu_z, log_sigma_z = self.encoder(x)
        sigma_z = torch.exp(log_sigma_z)
        mu_z = torch.exp(log_mu_z)
        estimated_reconstruction_likelihood = self.estimate_reconstruction_likelihood(
            x, log_mu_z, log_sigma_z, n_samples=n_samples
        )
        kl_divergence = -(
            0.5 * (torch.ones_like(log_mu_z) + 2 * log_mu_z - torch.pow(mu_z, 2) - torch.pow(sigma_z, 2))
        ).sum(dim=2)[
            0
        ]  # B x 1 x 1024 -> B

        return estimated_reconstruction_likelihood, kl_divergence

    def estimate_reconstruction_likelihood(self, x, log_mu_z, log_sigma_z, n_samples=1):

        log_probs = []
        for _ in range(n_samples):
            z = torch.unflatten(
                input=self.encoder_sampler.sample(log_mu=log_mu_z, log_sigma=log_sigma_z),
                dim=2,
                sizes=(int(self.latent_dim**0.5), -1),
            )
            log_mu_x, log_sigma_x = self.decoder(z)

            log_probs.append(
                self.decoder_sampler.log_prob(
                    log_mu=log_mu_x, log_sigma=log_sigma_x, x=torch.flatten(x, start_dim=2)
                )
            )  # B x 3 x pixel_dim

        probs = torch.stack(log_probs, dim=1)  # B x n_samples x 3 x pixel_dim
        probs = probs.sum(dim=[2, 3]).mean(dim=1)  # B
        return probs



class SimpleVariationalAutoEncoder(nn.Module):
    def __init__(self,):

        super().__init__()

        self.latent_dim = 200
        self.hidden_dim = 400
        self.pixel_dim = 784

        # encoder
        self.FC_input = nn.Linear(self.pixel_dim, self.hidden_dim)
        self.FC_input2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.FC_mean  = nn.Linear(self.hidden_dim, self.latent_dim)
        self.FC_var   = nn.Linear (self.hidden_dim, self.latent_dim)

        self.relu = nn.ReLU()

        # decoder
        self.FC_hidden = nn.Linear(self.latent_dim, self.hidden_dim)
        self.FC_hidden2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.FC_output_mean = nn.Linear(self.hidden_dim, self.pixel_dim)
        self.FC_output_var = nn.Linear(self.hidden_dim, self.pixel_dim)

        self.encoder_sampler = MultivariateNormal(dimension=self.latent_dim)
        self.decoder_sampler = MultivariateNormal(dimension=self.pixel_dim)

    def encoder(self,x):
        h_       = self.relu(self.FC_input(x))
        h_       = self.relu(self.FC_input2(h_))
        log_mean     = self.FC_mean(h_)
        log_var  = self.FC_var(h_)                                                
        return log_mean, log_var

    def decoder(self, z):
        h     = self.relu(self.FC_hidden(z))
        h     = self.relu(self.FC_hidden2(h))
        log_mean = self.FC_output_mean(h)
        log_var = self.FC_output_var(h)
        return log_mean, log_var


    def generate(self):

        z = torch.unflatten(
            input=self.encoder_sampler.sample_isotropic()[None],
            dim=1,
            sizes=(int(self.latent_dim**0.5), -1),
        )[:, None, :, :].to(next(self.parameters()).device)
        log_mu, log_sigma = self.decoder(z)
        flattened_x = self.decoder_sampler.sample(log_mu, log_sigma).to(next(self.parameters()).device)
        x = torch.unflatten(flattened_x, dim=2, sizes=(int(self.pixel_dim**0.5), -1))

        return x

    def compute_loss(self, x, n_samples=1):

        log_mu_z, log_sigma_z = self.encoder(x)
        sigma_z = torch.exp(log_sigma_z)
        mu_z = torch.exp(log_mu_z)
        estimated_reconstruction_likelihood = self.estimate_reconstruction_likelihood(
            x, log_mu_z, log_sigma_z, n_samples=n_samples
        )
        kl_divergence = -(
            0.5 * (torch.ones_like(log_mu_z) + 2 * log_sigma_z - torch.pow(mu_z, 2) - torch.pow(sigma_z, 2))
        ).sum(dim=2)[0]  # B x 1 x 1024 -> B

        return estimated_reconstruction_likelihood, kl_divergence

    def estimate_reconstruction_likelihood(self, x, log_mu_z, log_sigma_z, n_samples=1):

        log_probs = []
        for _ in range(n_samples):
            z = self.encoder_sampler.sample(log_mu=log_mu_z, log_sigma=log_sigma_z)
            log_mu_x, log_sigma_x = self.decoder(z)

            log_probs.append(
                self.decoder_sampler.log_prob(
                    log_mu=log_mu_x, log_sigma=log_sigma_x, x=x
                )
            )  # B x 3 x pixel_dim

        probs = torch.stack(log_probs, dim=1)  # B x n_samples x 3 x pixel_dim
        probs = probs.sum(dim=[2, 3]).mean(dim=1)  # B
        return probs



# vae = SimpleVariationalAutoEncoder()
# x = torch.ones((2,1,784))
# y = vae.compute_loss(x=x)


# vae = VariationalAutoEncoder(encoder_decoder_depth=3, encoder_start_channels=32)
# x = torch.ones((2, 3, 256, 256))
# y = vae.compute_loss(x=x)
# print(y)


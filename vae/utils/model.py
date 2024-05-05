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
    def __init__(self, in_channels_start, depth, in_channels=3):

        super().__init__()
        self.encoder_blocks = nn.ModuleList([])
        channels = in_channels_start

        self.encoder_blocks.append(ResidualEncoderBlock(in_channels=in_channels, out_channels=channels, residual=False))

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

        mu = x[:, [0], :]
        log_sigma = torch.clamp(x[:, [1], :], min=-20, max=2)

        return mu, log_sigma


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
    def __init__(self, in_channels_start, depth, out_channels=3):

        super().__init__()
        self.decoder_blocks = nn.ModuleList([])
        self.out_channels = out_channels
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
            nn.Conv2d(channels, 2*out_channels, 1)
        )  # 6 output channels: one per RGB channel, and for each i have log_mu and log_sigma
        self.decoder_blocks.append(nn.Flatten(start_dim=2))  # flatten the feature maps to B x 6 x latent_dim

    def forward(self, x):

        for block in self.decoder_blocks:
            x = block(x)

        mu = torch.nn.functional.sigmoid(x[:, :self.out_channels, :])
        log_sigma = torch.clamp(x[:, self.out_channels:, :], min=-20, max=2)

        return mu, log_sigma


class VariationalAutoEncoder(nn.Module):
    def __init__(self, encoder_decoder_depth, encoder_start_channels, img_dim = 256, img_channels=3):

        super().__init__()

        self.encoder = ResidualEncoder(in_channels_start=encoder_start_channels, depth=encoder_decoder_depth, in_channels=img_channels)
        self.decoder = ResidualDecoder(
            in_channels_start=int(encoder_start_channels * (2) ** encoder_decoder_depth),
            depth=encoder_decoder_depth, out_channels=img_channels,
        )
        self.latent_dim = int(img_dim * 0.5** encoder_decoder_depth) ** 2
        self.pixel_dim = img_dim * img_dim
        self.encoder_sampler = MultivariateNormal(dimension=self.latent_dim)
        self.decoder_sampler = MultivariateNormal(dimension=self.pixel_dim)

        self.activations = {module_name: [] for module_name, module in self.named_modules()}
        # self.register_custom_hooks()


    def register_custom_hooks(self):

        for module_name, module in self.named_modules():

            def module_hook(_, __, outputs, module_name=module_name):
                if isinstance(outputs, tuple):
                    for i,output in enumerate(outputs):
                        module_activation_name = f'{module_name}_{i}'
                        if module_activation_name not in self.activations:
                            self.activations[module_activation_name] = []
                        self.activations[module_activation_name].append(output.detach().cpu())
                else:
                    self.activations[module_name].append(outputs.detach().cpu())
            module.register_forward_hook(module_hook)


    def generate(self, x=None):

        if x is not None:
            mu_z, log_sigma_z = self.encoder(x)
            z = torch.unflatten(
                input=self.encoder_sampler.sample(mu=mu_z, log_sigma=log_sigma_z),
                dim=2,
                sizes=(int(self.latent_dim**0.5), -1),
            )  
        else:
            z = self.encoder_sampler.sample_isotropic().to(next(self.parameters()).device)[:,None, :, :]
        mean, log_sigma = self.decoder(z)
        flattened_x = self.decoder_sampler.sample(mean, log_sigma).to(next(self.parameters()).device)
        x = torch.unflatten(mean, dim=2, sizes=(int(self.pixel_dim**0.5), -1))

        return x

    def compute_loss(self, x, n_samples=1, mse=False):


        mu_z, log_sigma_z = self.encoder(x)
        sigma_z = torch.exp(log_sigma_z)
        
        if mse:
            mse_loss = nn.MSELoss()
            z = torch.unflatten(
                input=self.encoder_sampler.sample(mu=mu_z, log_sigma=log_sigma_z),
                dim=2,
                sizes=(int(self.latent_dim**0.5), -1),
            )            
            y, _ = self.decoder(z)
            y = y.unflatten(dim=2, sizes=(int(y.shape[2]**0.5), -1))
            estimated_reconstruction_likelihood = - mse_loss(x, y)
        else:
            estimated_reconstruction_likelihood = self.estimate_reconstruction_likelihood(
                x, mu_z, log_sigma_z, n_samples=n_samples
            )
        kl_divergence = -(
            0.5 * (torch.ones_like(mu_z) + 2 * log_sigma_z - torch.pow(mu_z, 2) - torch.pow(sigma_z, 2))
        ).sum(dim=2)[:,0]  # B x 1 x 1024 -> B

        return estimated_reconstruction_likelihood, kl_divergence

    def estimate_reconstruction_likelihood(self, x, mu_z, log_sigma_z, n_samples=1):

        log_probs = []
        for _ in range(n_samples):
            z = torch.unflatten(
                input=self.encoder_sampler.sample(mu=mu_z, log_sigma=log_sigma_z),
                dim=2,
                sizes=(int(self.latent_dim**0.5), -1),
            )            
            mu_x, log_sigma_x = self.decoder(z)

            log_probs.append(
                self.decoder_sampler.log_prob(
                    mu=mu_x, log_sigma=log_sigma_x, x=x.flatten(start_dim=2)
                )
            )  # B x 3 x pixel_dim

        probs = torch.stack(log_probs, dim=1)  # B x n_samples x 3 x pixel_dim
        probs = probs.sum(dim=[2, 3]).mean(dim=1)  # B
        probs = torch.clamp(probs, min=torch.quantile(probs, 0.10), max = torch.quantile(probs, 0.999))
        return probs



class SimpleVariationalAutoEncoder(nn.Module):
    def __init__(self,):

        super().__init__()

        self.latent_dim = 64
        self.hidden_dim = 128
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

        # self.apply(self.weight_initialiser())

        self.encoder_sampler = MultivariateNormal(dimension=self.latent_dim)
        self.decoder_sampler = MultivariateNormal(dimension=self.pixel_dim)

        self.activations = {module_name: [] for module_name, module in self.named_modules()}
        self.register_custom_hooks()

    def weight_initialiser(self):

        def f(m):

            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, gain=0.1)
                m.bias.data.fill_(0.01)
        return f

    def register_custom_hooks(self):

        for module_name, module in self.named_modules():

            def module_hook(_, __, output, module_name=module_name):
                self.activations[module_name].append(output.detach().cpu())
            module.register_forward_hook(module_hook)

    def encoder(self,x):
        h_       = self.relu(self.FC_input(x))
        h_       = self.relu(self.FC_input2(h_))
        mean     = self.FC_mean(h_)
        log_var  = torch.clamp(self.FC_var(h_), min=-20, max=1)
        return mean, log_var

    def decoder(self, z):
        h     = self.relu(self.FC_hidden(z))
        h     = self.relu(self.FC_hidden2(h))
        mean = torch.nn.functional.sigmoid(self.FC_output_mean(h))
        log_var = torch.clamp(self.FC_output_var(h), min=-10, max=1)

        return mean, log_var


    def generate(self, x=None):

        if x is not None:
            mu_z, log_sigma_z = self.encoder(x)
            z = self.encoder_sampler.sample(mu=mu_z, log_sigma=log_sigma_z)
        else:
            z = self.encoder_sampler.sample_isotropic().to(next(self.parameters()).device)[None, None, :]
        mean, log_sigma = self.decoder(z)
        flattened_x = self.decoder_sampler.sample(mean, log_sigma).to(next(self.parameters()).device)
        x = torch.unflatten(mean, dim=2, sizes=(int(self.pixel_dim**0.5), -1))

        return x

    def compute_loss(self, x, n_samples=1, simple=False):

        if simple:
            mu_z, log_sigma_z = self.encoder(x)
            z = self.encoder_sampler.sample(mu=mu_z, log_sigma=log_sigma_z)
            mu_x, log_sigma_x = self.decoder(z)
            reproduction_loss = nn.functional.binary_cross_entropy(mu_x, x, reduction='sum')
            KLD      = - 0.5 * torch.sum(1+ 2*log_sigma_z - mu_z.pow(2) - (2*log_sigma_z).exp())
            return - reproduction_loss, KLD
        
        else:
            mu_z, log_sigma_z = self.encoder(x)
            sigma_z = torch.exp(log_sigma_z)
            estimated_reconstruction_likelihood = self.estimate_reconstruction_likelihood(
                x, mu_z, log_sigma_z, n_samples=n_samples
            )
            kl_divergence = -(
                0.5 * (torch.ones_like(mu_z) + 2 * log_sigma_z - torch.pow(mu_z, 2) - torch.pow(sigma_z, 2))
            ).sum(dim=2)[:,0]  # B x 1 x 1024 -> B

            return estimated_reconstruction_likelihood, kl_divergence

    def estimate_reconstruction_likelihood(self, x, mu_z, log_sigma_z, n_samples=1):

        log_probs = []
        for _ in range(n_samples):
            z = self.encoder_sampler.sample(mu=mu_z, log_sigma=log_sigma_z)
            mu_x, log_sigma_x = self.decoder(z)

            log_probs.append(
                self.decoder_sampler.log_prob(
                    mu=mu_x, log_sigma=log_sigma_x, x=x
                )
            )  # B x 3 x pixel_dim

        probs = torch.stack(log_probs, dim=1)  # B x n_samples x 3 x pixel_dim
        probs = probs.sum(dim=[2, 3]).mean(dim=1)  # B
        probs = torch.clamp(probs, min=-2000, max = 20000)
        return probs


# vae = SimpleVariationalAutoEncoder()
# x = torch.ones((2,1,784))
# y = vae.compute_loss(x=x)


# vae = VariationalAutoEncoder(encoder_decoder_depth=3, encoder_start_channels=32)
# x = torch.ones((2, 3, 256, 256))
# y = vae.compute_loss(x=x)
# print(y)


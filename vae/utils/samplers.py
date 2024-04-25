import torch


class MultivariateNormal:
    def __init__(self, dimension):

        self.dimension = dimension
        self.normal = torch.distributions.MultivariateNormal(
            loc=torch.zeros(dimension), scale_tril=torch.eye(n=dimension)
        )

    def sample(self, log_mu, log_sigma):

        samples = self.normal.sample((log_mu.shape[0:2]))

        return torch.exp(log_mu) + torch.exp(log_sigma) * samples[0]

    def sample_isotropic(self):

        return self.normal.sample()

    def log_prob(self, log_mu, log_sigma, x):

        mu = torch.exp(log_mu)
        sigma_square = torch.pow(torch.exp(log_sigma), 2)
        sigma_square = torch.diag_embed(sigma_square[..., None].repeat_interleave(self.dimension, dim=-1))

        mu = mu.flatten(end_dim=1)  # flatten the filter and batch in one size due to gaussian implementation
        sigma_square = sigma_square.flatten(end_dim=1)

        gaussian = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=sigma_square[:, 0, :, :])
        return gaussian.log_prob(x.flatten(end_dim=1))

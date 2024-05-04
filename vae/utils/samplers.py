import torch


class MultivariateNormal:
    def __init__(self, dimension):

        self.dimension = dimension
        self.normal = torch.distributions.MultivariateNormal(
            loc=torch.zeros(dimension), scale_tril=torch.eye(n=dimension)
        )

    def sample(self, log_mu, log_sigma):

        samples = self.normal.sample((log_mu.shape[0:2]))
        samples = samples.to(log_mu.device)

        return torch.exp(log_mu) + torch.exp(log_sigma) * samples[0]

    def sample_isotropic(self):

        return self.normal.sample()

    def log_prob(self, log_mu, log_sigma, x):

        original_shape = log_mu.shape
        mu = torch.exp(log_mu)
        sigma = torch.exp(log_sigma) + 0.001 * torch.ones_like(log_sigma)

        mu = mu.flatten()
        sigma = sigma.flatten()

        gaussian = torch.distributions.normal.Normal(loc=mu, scale=sigma)
        return gaussian.log_prob(x.flatten()).unflatten(dim=0, sizes=original_shape)

import torch


class MultivariateNormal:
    def __init__(self, dimension):

        self.dimension = dimension
        self.normal = torch.distributions.MultivariateNormal(
            loc=torch.zeros(dimension), scale_tril=torch.eye(n=dimension)
        )

    def sample(self, mu, log_sigma):

        samples = self.normal.sample((mu.shape[0:2]))
        samples = samples.to(mu.device)

        return mu + torch.exp(log_sigma) * samples

    def sample_isotropic(self):

        return self.normal.sample()

    def log_prob(self, mu, log_sigma, x):

        original_shape = mu.shape
        sigma = torch.exp(log_sigma) + 0.000001 * torch.ones_like(log_sigma)

        mu = mu.flatten()
        sigma = sigma.flatten()

        gaussian = torch.distributions.normal.Normal(loc=mu, scale=sigma)
        return gaussian.log_prob(x.flatten()).unflatten(dim=0, sizes=original_shape)


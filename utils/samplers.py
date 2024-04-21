import torch


class MultivariateNormal:
    def __init__(self, dimension):

        self.normal = torch.distributions.MultivariateNormal(
            loc=torch.zeros(dimension), scale_tril=torch.eye(n=dimension)
        )

    def sample(self, log_mu, log_sigma):

        samples = self.normal.sample((log_mu.shape[0:2]))

        return torch.exp(log_mu) + torch.exp(log_sigma) * samples[0]

    def sample_isotropic(self):

        return self.normal.sample()

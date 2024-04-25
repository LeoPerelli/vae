from vae.utils.samplers import MultivariateNormal
import torch


def test_MultiVariateNormal():

    s = MultivariateNormal(4)
    log_sigma = torch.ones((2, 3, 4))
    log_mu = torch.ones((2, 3, 4))
    x = torch.ones((2, 3, 4))
    s.log_prob(log_mu, log_sigma, x)

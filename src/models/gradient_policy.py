import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class GradientPolicy(nn.Module):
    def __init__(self, hidden_size, obs_size, out_dims, max, device):
        super().__init__()
        self.device = device
        self.max = torch.from_numpy(max).to(device)
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(hidden_size, out_dims)
        self.log_std_layer = nn.Linear(hidden_size, out_dims)

    def forward(self, obs, deterministic=False, with_logprob=True):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(self.device)
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        if deterministic:
            action = mu
        else:
            action = pi_distribution.rsample()

        if with_logprob:
            log_prob = pi_distribution.log_prob(action)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            log_prob -= (
                2 * (np.log(2) - action - F.softplus(-2 * action))
            ).sum(dim=-1, keepdim=True)
        else:
            log_prob = None

        action = torch.tanh(action)
        action = self.max * action

        return action, log_prob

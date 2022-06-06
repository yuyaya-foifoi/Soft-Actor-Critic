import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


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
        self.linear_mu = nn.Linear(hidden_size, out_dims)
        self.linear_std = nn.Linear(hidden_size, out_dims)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(self.device)
        x = self.net(obs.float())
        mu = self.linear_mu(x)
        std = self.linear_std(x)
        std = F.softplus(std) + 1e-3

        dist = Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        log_prob -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(
            dim=-1, keepdim=True
        )

        action = torch.tanh(action) * self.max
        return action, log_prob

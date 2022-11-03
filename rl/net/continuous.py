import torch
from torch import nn
from torch.distributions import Independent, Normal
from typing import Tuple

from rl.net.common import Net

MIN_SIGMA = -20
MAX_SIGMA = 2


class Critic(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_layers: Tuple[int, ...] = (128,),
            activation: str = 'tanh'
    ) -> None:
        super().__init__()
        self.preprocess = Net(input_dim, 0, hidden_layers, activation)
        self.value = nn.Linear(hidden_layers[-1], 1)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.value(self.preprocess(s))


class Actor(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_layers: Tuple[int, ...] = 128,
            activation: str = 'tanh',
            conditioned_sigma: bool = False,
            deploy: bool = False
    ) -> None:
        super().__init__()
        self.preprocess = Net(input_dim, 0, hidden_layers, activation)
        self.mu = nn.Linear(hidden_layers[-1], output_dim)
        # self.mu = SplitNet(hidden_layers[-1], (4, 12))
        self._c_sigma = conditioned_sigma
        self.deploy = deploy
        if conditioned_sigma:
            self.sigma = nn.Linear(hidden_layers[-1], output_dim)
        else:
            self.sigma = nn.Parameter(torch.ones(1, output_dim) * -0.1)
        self.dist_fn = lambda *logits: Independent(Normal(*logits), 1)

    def forward(self, s: torch.Tensor):
        layer2_out = self.preprocess(s)
        mu = torch.tanh(self.mu(layer2_out))
        if self.deploy: return mu
        if self._c_sigma:
            sigma = torch.clamp(self.sigma, min=MIN_SIGMA, max=MAX_SIGMA).exp()
        else:
            sigma = (self.sigma + torch.zeros_like(mu)).exp()
        logits = mu, sigma
        if self.training:
            dist = self.dist_fn(*logits)
            return {'logits': logits, 'dist': dist, 'act': dist.sample()}
        else:
            return {'logits': logits, 'act': mu}


class SplitNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: Tuple[int, int]):
        super(SplitNet, self).__init__()
        self.nets = nn.ModuleList([nn.Linear(input_dim, dim) for dim in output_dim])

    def forward(self, s: torch.Tensor):
        out = [net(s) for net in self.nets]
        return torch.cat(out, dim=-1)

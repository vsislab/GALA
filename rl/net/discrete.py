import torch
import torch.nn.functional as F
from torch import nn
from typing import Tuple

from rl.net.common import Net
from .continuous import Critic


class Actor(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_layers: Tuple[int, ...] = (128,),
            activation: str = 'tanh',
    ) -> None:
        super().__init__()
        self.preprocess = Net(input_dim, 0, hidden_layers, activation)
        self.prob = nn.Linear(hidden_layers[-1], output_dim)
        self.dist_fn = torch.distributions.Categorical

    def forward(self, s: torch.Tensor) -> dict:
        net_out = self.preprocess(s)
        logits = F.softmax(self.prob(net_out), dim=-1)
        if self.training:
            dist = self.dist_fn(logits)
            return {'logits': logits, 'dist': dist, 'act': dist.sample()}
        else:
            return {'logits': logits, 'act': logits.argmax(-1)}


# class DeployActor(Actor):
#     def forward(self, s):
#         net_out = self.preprocess(s)
#         return F.softmax(self.prob(net_out), dim=-1).argmax(-1)

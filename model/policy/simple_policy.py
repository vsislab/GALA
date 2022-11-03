# -*- coding: utf-8 -*-
import numpy as np
import torch
from typing import Union
import rl


class Actor(rl.ContinuousActor):
    def __init__(self, state_dim: int, action_dim: int, device: str = 'cpu', deploy=False, **kwargs):
        super(Actor, self).__init__(state_dim, action_dim, (256, 128), activation='relu', deploy=deploy)
        self.device = device

    def forward(self, s: Union[np.ndarray, torch.Tensor]):
        s = rl.to_torch(s, device=self.device, dtype=torch.float32)
        return super(Actor, self).forward(s)


class Critic(rl.ContinuousCritic):
    def __init__(self, state_dim: int, device: str = 'cpu', **kwargs):
        super(Critic, self).__init__(state_dim, (256, 128), activation='relu')
        self.device = device

    def forward(self, s: Union[np.ndarray, torch.Tensor]):
        s = rl.to_torch(s, device=self.device, dtype=torch.float32)
        return super(Critic, self).forward(s)


if __name__ == '__main__':
    c = Critic(10)
    o = c(torch.zeros(0, 10))
    x = torch.ones(10, 1)
    c(o, x=o)

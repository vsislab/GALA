# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from typing import Union, Tuple
import rl


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, device: str = 'cpu', deploy=False, **kwargs):
        super(Actor, self).__init__()
        self.policy_net = rl.ContinuousActor(state_dim, action_dim, (256, 256), activation='relu', deploy=deploy)
        self.device = device

    def forward(self, s: Union[np.ndarray, torch.Tensor]):
        s = rl.to_torch(s, device=self.device, dtype=torch.float32)
        return self.policy_net(s)


class Critic(nn.Module):
    def __init__(self, state_dim: int, task_dim: int, task_adv_coef: list = None, device: str = 'cpu', **kwargs):
        super(Critic, self).__init__()
        self.task_dim = task_dim
        if task_adv_coef is not None:
            self.task_adv_coef = np.asarray(task_adv_coef)
            assert all(self.task_adv_coef >= 0.) and all(self.task_adv_coef <= 1.)
        else:
            self.task_adv_coef = np.ones(task_dim) * 1
        self.device = device
        self.value_nets = nn.ModuleList([])
        for i in range(task_dim):
            value_net = rl.ContinuousCritic(state_dim, (256, 256), activation='relu')
            self.value_nets.append(value_net)

    def forward(self, s: Union[np.ndarray, torch.Tensor], net_id: int = None):
        s = rl.to_torch(s, device=self.device, dtype=torch.float32)
        return self.value_nets[net_id](s)

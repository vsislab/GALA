# -*- coding: utf-8 -*-
import torch


class BaseActor(torch.nn.Module):
    def __init__(self, state_dim: int, action_dim: int, device: str = 'cpu'):
        super(BaseActor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device


class BaseCritic(torch.nn.Module):
    def __init__(self, state_dim: int, device: str = 'cpu'):
        super(BaseCritic, self).__init__()
        self.state_dim = state_dim
        self.device = device

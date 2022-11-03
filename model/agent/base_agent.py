# -*- coding: utf-8 -*-
import numpy as np
import torch
from typing import Any


class BaseAgent:
    def __init__(self, actor: torch.nn.Module):
        self.actor = actor

    def reset(self, obs: np.ndarray):
        pass

    def __call__(self, obs: np.ndarray) -> Any:
        raise NotImplementedError

    def share_memory(self):
        self.actor.share_memory()

    def to_device(self, device: str = 'cpu'):
        self.actor.to(device)
        self.actor.device = device

    # def train(self, mode=True):
    #     self.training = mode
    #
    # def eval(self):
    #     self.train(False)

# -*- coding: utf-8 -*-
import numpy as np
import torch

from .base_agent import BaseAgent


class SimpleAgent(BaseAgent):
    def __init__(self, actor: torch.nn.Module):
        super(SimpleAgent, self).__init__(actor)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            act = self.actor(obs[None])['act'].cpu().numpy()[0]
        return act

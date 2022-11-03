# -*- coding: utf-8 -*-
import copy
import importlib
import torch
import torch.nn as nn

from .agent import SimpleAgent


def load_agent(actor: nn.Module):
    agent = SimpleAgent(actor)
    return agent



def load_actor(cfg: dict, device='cpu', deploy=False, **kwargs):
    assert 'name' in cfg
    cfg = copy.deepcopy(cfg)
    policy_module = importlib.import_module(f"model.policy.{cfg['name']}")
    del cfg['name']
    return getattr(policy_module, 'Actor')(**cfg, device=device, deploy=deploy, **kwargs).to(device)


def load_critic(cfg: dict, device='cpu', **kwargs):
    assert 'name' in cfg
    cfg = copy.deepcopy(cfg)
    policy_module = importlib.import_module(f"model.policy.{cfg['name']}")
    del cfg['name']
    return getattr(policy_module, 'Critic')(**cfg, device=device, **kwargs).to(device)


def orthogonal_linear_weights(net: nn.Module):
    """orthogonal initialization"""
    for m in list(net.modules()):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)

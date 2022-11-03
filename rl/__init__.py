# -*- coding: utf-8 -*-
from .net.continuous import Actor as ContinuousActor
from .net.continuous import Critic as ContinuousCritic
from .net.discrete import Actor as DiscreteActor
from .net.discrete import Critic as DiscreteCritic
from .net.common import Net
from .data import to_numpy, to_torch, to_torch_as

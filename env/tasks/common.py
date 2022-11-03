# -*- coding: utf-8 -*-
import gym
from math import e, exp
import numpy as np
from typing import Dict

TASK = {}


def is_valid_task(name: str):
    return name in TASK


def register(cls):
    cls.name = cls.__name__
    assert cls.name not in TASK, cls.name
    TASK[cls.name] = cls
    return cls


def check(cls):
    assert any([cls == v for v in TASK.values()]), \
        f"Please register the '{cls.name}'"


def load_task_cls(cfg: Dict):
    name = cfg['name']
    if name in TASK:
        task_cls = TASK[name]
        task_cls.configure(cfg)
        return task_cls
    else:
        raise KeyError(f'Not exist task named {name}.')


def build_discrete_space(n):
    return gym.spaces.Discrete(n)


def build_box_space(low, high):
    low = np.array(low, dtype=np.float32)
    high = np.array(high, dtype=np.float32)
    assert low.size == high.size and all(high > low)
    return gym.spaces.Box(low, high)


def G(x):
    return 1 / (e ** x + e ** (-x) + 2)


def C(x):
    return -1 / (e ** x + e ** (-x) + 2) + 0.25


def K(x):
    return 1 / (4 ** x + 4 ** (-x) + 2)


def M(x):
    return 1 / (1.4 ** x + 1.4 ** (-x) + 2)


def S(x):
    return x ** 0.5


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.linspace(-1, 1, 1000)
    y = S(x)
    plt.plot(x, y)
    plt.show()

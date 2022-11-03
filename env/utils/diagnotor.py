# -*- coding: utf-8 -*-
import collections
import numpy as np


class Diagnotor:
    def __init__(self, env, task, size: int = 1000, diagnostic_step: int = 100, diagnostic_prob: float = 0.8):
        self.env = env
        self.task = task
        self.size = size
        self.diagnostic_prob = diagnostic_prob
        self.diagnostic_step = diagnostic_step
        self.tracing_states = []
        self.diagnostic_states = collections.deque(maxlen=1000)
        self.reset()

    def reset(self):
        self.tracing_states.clear()
        self.diagnostic_states.clear()

    def trace(self, end_flag, early_stop):
        self.tracing_states.append({**self.env.aliengo.state_dict, **self.task.state_dict})
        if end_flag:
            # if early_stop:
            # self.diagnostic_states.extend(self.tracing_states[-self.diagnostic_step:])
            self.diagnostic_states.extend(self.tracing_states[::10])
            self.tracing_states = []

    def pop(self) -> dict:
        p = self.diagnostic_prob * float(len(self.diagnostic_states) > 0)
        if np.random.choice([True, False], p=[p, 1 - p]):
            return np.random.choice(self.diagnostic_states)
        else:
            return {}

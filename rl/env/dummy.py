# -*- coding: utf-8 -*-
import gym
import multiprocessing as mp
import numpy as np
import warnings
from typing import List, Any

from model.agent import BaseAgent


def any_stack(l: List[Any]):
    if isinstance(l[0], dict):
        return {key: any_stack([x[key] for x in l]) for key in l[0].keys()}
    else:
        return np.stack(l)


class ReplayBuffer:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._index = 0
        self.meta = {'obs': [], 'act': [], 'rew': [], 'obs_next': [], 'done': [], 'info': []}

    def add(self, obs, act, rew, obs_next, done, info):
        self.meta['obs'].append(obs)
        self.meta['act'].append(act)
        self.meta['rew'].append(rew)
        self.meta['obs_next'].append(obs_next)
        self.meta['done'].append(done)
        self.meta['info'].append(info)
        self._index += 1

    @property
    def data(self):
        if len(self) == 0:
            warnings.warn('Caution: the length of the replay buffer is 0.')
            return self._meta
        return {
            'obs': any_stack(self.meta['obs']),
            'act': any_stack(self.meta['act']),
            'rew': any_stack(self.meta['rew']),
            'obs_next': any_stack(self.meta['obs_next']),
            'done': any_stack(self.meta['done']),
            'info': any_stack(self.meta['info']),
        }

    def __getattr__(self, name):
        return self.meta[name]

    def __len__(self) -> int:
        return self._index


class Collector:
    def __init__(self, env: gym.Env, agent: BaseAgent):
        self.env = env
        self.agent = agent
        self.buffer = ReplayBuffer()

    def reset(self) -> None:
        obs = self.env.reset()
        self.agent.reset(obs)
        self.buffer.reset()
        return obs

    def collect(self, traffic_signal: mp.Value):
        # start_time = time.time()
        obs = self.reset()
        while True:
            act = self.agent(obs)
            obs_next, rew, done, info = self.env.step(act)
            done = done or (not traffic_signal.value)
            self.buffer.add(obs, act, rew, obs_next, done, info)
            if done: break
            obs = obs_next
        # duration = max(time.time() - start_time, 1e-9)
        # print(f'{os.getpid()} : {round(duration, 2)}')
        return {
            'len': len(self.buffer),
            'rew': sum(self.buffer.rew),
            'suc': info['success'],
            # 'time': duration
        }

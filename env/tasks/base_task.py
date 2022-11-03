# -*- coding: utf-8 -*-
# import gym
import numpy as np
from typing import List, Dict, Tuple

from .common import check, build_box_space


class BaseTask:
    name = None
    reward_name = []
    config = {'action_low': None, 'action_high': None}

    def __init__(self, env):
        check(self.__class__)
        self.env = env
        self.schedule_ratio = 1.
        self.debug = False
        self._debug_param = {}
        self.build_action_space()
        self.build_observation_space()

    @classmethod
    def configure(cls, cfg: Dict):
        # for k in ['action_low', 'action_high']:
        #     if k in cfg:
        #         setattr(cls, k, cfg[k])
        # for k in cls.config:
        #     if k in cfg:
        #         cls.config[k] = cfg[k]
        cls.config.update(cfg)

    def enable_debug(self, mode=True):
        self.debug = mode

    def transform(self, net_out):
        net_out = np.clip(net_out, -1., 1.)
        net_out = (net_out + 1) / 2 * (self.action_space.high - self.action_space.low) + self.action_space.low
        return net_out

    def reset(self, **kwargs):
        """This is called after environment reset"""
        raise NotImplementedError

    def observation(self, obs):
        raise NotImplementedError

    def action(self, net_out):
        raise NotImplementedError

    def reward(self) -> List:
        """This is called after stepSimulation"""
        raise NotImplementedError

    def terminate(self) -> Tuple[bool, dict]:
        """This is called after stepSimulation"""
        raise NotImplementedError

    # def info(self):
    #     return {'success': 0.}

    def refresh_observation_noise(self, support_mask):
        default_noise = self.env.aliengo.default_observation_noise
        noise = self.env.aliengo.observation_noise
        velocity_noise, rpy_noise, rpy_rate_noise = default_noise['velocity'], default_noise['rpy'], default_noise['rpy_rate']
        if sum(support_mask) < 2:
            velocity_noise *= 8.
            rpy_noise *= 8.
            rpy_rate_noise *= 8.
        noise['velocity'] = velocity_noise
        noise['rpy'] = rpy_noise
        noise['rpy_rate'] = rpy_rate_noise

    def refresh(self, **kwargs):
        raise NotImplementedError

    def get_reset_state(self) -> dict:
        return {}

    @property
    def debug_param(self) -> Dict:
        return self._debug_param

    @property
    def debug_name(self) -> Dict:
        return {}

    def build_observation_space(self):
        self.get_reset_state()
        self.reset()
        obs = self.observation(self.env.aliengo.state_dict)
        observation_scale = np.array([float('inf')] * len(obs), dtype=np.float32)
        self.observation_space = build_box_space(-observation_scale, observation_scale)

    def build_action_space(self):
        self.action_space = build_box_space(self.config['action_low'], self.config['action_high'])

    @property
    def task_dim(self):
        return 1

    def __repr__(self):
        return self.name


class BaseMCTask(BaseTask):
    task_cls: List
    task_adv_coef: List = None

    def __init__(self, env):
        super(BaseMCTask, self).__init__(env)
        self.task_reward_fn = [t.reward for t in self.task_cls]
        self.task_reward_name = [t.reward_name for t in self.task_cls]

    @property
    def task_dim(self):
        return len(self.task_cls)

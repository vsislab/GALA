# -*- coding: utf-8 -*-
import copy
import gym
import numpy as np
import pandas as pd
import sys
import warnings
from typing import Union

from .aliengo_gym_env import AliengoGymEnv
from .tasks.base_task import BaseTask, BaseMCTask


class AliengoGymEnvWrapper(gym.Wrapper):
    registered_functions = []

    def __init__(self, env: AliengoGymEnv, task: Union[BaseTask, BaseMCTask], debug: bool = False):
        super(AliengoGymEnvWrapper, self).__init__(env)
        self._is_multi_critic_task = task.task_dim > 1
        self.task = task
        self.observation_space = task.observation_space
        self.action_space = task.action_space
        self.debug = debug
        self.task.enable_debug(debug)
        self.debug_data = None
        self.action = None

    def reset(self, **kwargs):
        for i in range(100):
            reset_state = self.task.get_reset_state()
            reset_state.update(kwargs)
            obs = self.env.reset(**reset_state)
            if obs is not None:
                break
            if i >= 50:
                warnings.warn('Too much reset!\n'
                              f'self collision: {self.env.aliengo.self_collision}\n'
                              # f'terrain penetration：{self.env.terrain_penetration}'
                              )
                self.env.close()
                sys.exit(-1)
        self.task.reset()
        if self.debug:
            self.debug_data = {
                'action': [self.env.aliengo.init_motor_position],
                'reward': [np.zeros(len(self.task.reward_name))],
                'done': [[0]],
                'base_velocity': [np.zeros(3)],
                'base_rpy_rate': [np.zeros(3)],
                'base_foot_position': [self.env.aliengo.get_foot_position_on_base_frame().transpose().flatten()],
                'foot_position': [self.env.aliengo.foot_position.transpose().flatten()],
                'foot_velocity': [self.env.aliengo.foot_velocity.transpose().flatten()]
            }
            if self._is_multi_critic_task:
                for i in range(self.task.task_dim):
                    self.debug_data[f'R{i}'] = [np.zeros(len(self.task.task_reward_name[i]))]
        return self.observation(obs)

    def step(self, act):
        self.action = action = self.task.action(act)
        obs, _, done, info = self.env.step(action)
        self.task.refresh()
        reward = self.task.reward()
        task_done, task_info = self.task.terminate()
        done = done or task_done
        info.update(task_info)
        if self.debug:
            if self._is_multi_critic_task:
                for i in range(self.task.task_dim):
                    self.debug_data[f'R{i}'].append(np.asarray(reward) * 100 if info['task_id'] == i else np.zeros(
                        len(self.task.task_reward_name[i])))
            else:
                self.debug_data['reward'].append(np.asarray(reward) * 100)
            self.debug_data['action'].append(action)
            self.debug_data['done'].append([float(done)])
            self.debug_data['base_velocity'].append(self.env.aliengo.base_velocity)
            self.debug_data['base_rpy_rate'].append(self.env.aliengo.base_rpy_rate)
            self.debug_data['base_foot_position'].append(
                self.env.aliengo.get_foot_position_on_base_frame().transpose().flatten())
            self.debug_data['foot_position'].append(self.env.aliengo.foot_position.transpose().flatten())
            self.debug_data['foot_velocity'].append(self.env.aliengo.foot_velocity.transpose().flatten())
        # if self._is_multi_critic_task:
        #     reward = np.asarray([sum(r) for r in reward])
        # else:
        reward = sum(reward)
        return self.observation(obs), reward, done, info

    def close(self):
        return super(AliengoGymEnvWrapper, self).close()

    def action(self):
        return self.action

    def observation(self, obs):
        if self.debug:
            obs_debug = copy.deepcopy(obs)
            for key, value in obs_debug.items():
                if key not in self.debug_data:
                    self.debug_data[key] = []
                self.debug_data[key].append(value)
            for key, value in self.task.debug_param.items():
                if key not in self.debug_data:
                    self.debug_data[key] = []
                self.debug_data[key].append(value)
        return self.task.observation(obs)

    def save_debug_report(self, path: str):
        if self.debug and self.debug_data is not None:
            with pd.ExcelWriter(path) as f:
                for key in self.debug_order:
                    # pd.DataFrame(np.asarray(self._debug_data[key]), columns=self.debug_name[key]).to_excel(f, key, float_format='%.3f')
                    pd.DataFrame(np.asarray(self.debug_data[key]), columns=self.debug_name[key]).to_excel(f, key,
                                                                                                          index=False)
            # print(f'The debug report has been written into `{path}`.')
            self.debug_data = None

    def save_action_report(self, path: str):
        if self.debug and self.debug_data is not None:
            with pd.ExcelWriter(path) as f:
                pd.DataFrame(np.asarray(self.debug_data['action']), columns=self.debug_name['action']).to_excel(f,
                                                                                                                'action',
                                                                                                                index=False)
            self.debug_data = None

    @property
    def debug_name(self):
        d = {**self.env.debug_name,
             **self.task.debug_name,
             'reward': self.task.reward_name,
             'done': ['done'],
             'base_velocity': ['x', 'y', 'z'],
             'base_rpy_rate': ['x', 'y', 'z'],
             'base_foot_position': [f'{l}_{o}' for o in ['x', 'y', 'z'] for l in self.env.aliengo.leg_names],
             'foot_position': [f'{l}_{o}' for o in ['x', 'y', 'z'] for l in self.env.aliengo.leg_names],
             'foot_velocity': [f'{l}_{o}' for o in ['x', 'y', 'z'] for l in self.env.aliengo.leg_names]}
        if self._is_multi_critic_task:
            d.update({f'R{i}': self.task.task_reward_name[i] for i in range(self.task.task_dim)})
        return d

    @property
    def debug_order(self):
        l = [f'R{i}' for i in range(self.task.task_dim)] if self._is_multi_critic_task else []
        l.extend([
            'reward',
            'base_velocity',
            'base_rpy_rate',
            *list(self.task.debug_name),
            *list(self.env.debug_name),
            'base_foot_position',
            'foot_position',
            'foot_velocity',
            'done'])
        return l

    @property
    def is_multi_critic_task(self):
        return self._is_multi_critic_task

    @classmethod
    def register(cls, f):
        cls.registered_functions.append(f.__name__)
        setattr(cls, f.__name__, f)
        return f

    def callback(self, func_name: str, params: dict = None):
        if not hasattr(self, func_name):
            raise NameError(f"You maybe forget to register the function '{func_name}' "
                            f"or register it inside a running function.")
        if params is None:
            params = {}
        return getattr(self, func_name)(**params)


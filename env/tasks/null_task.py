# -*- coding: utf-8 -*-
import numpy as np
from math import pi
from .base_task import BaseTask
from .common import register


@register
class NullTask(BaseTask):
    MOTOR_POSITION_HIGH = np.array([1.222, pi, -0.646]).repeat(4)
    MOTOR_POSITION_LOW = np.array([-1.222, -pi, -2.775]).repeat(4)

    config = {'base_frequency': 0., 'incremental': True, 'motor_position': True}
    config['action_high'] = np.array([1.5, 0.1, 0.15, 0.12]).repeat(4)  # residual:(f, Vx, Vy, Vz)
    config['action_low'] = np.array([-1.5, -0.1, -0.15, 0.0]).repeat(4)

    # action_high = np.array([1.], dtype=np.float32)
    # action_low = -action_high

    def __init__(self, env, **kwargs):
        super(NullTask, self).__init__(env)

    def reset(self):
        return False

    def refresh(self, reset=False):
        pass

    def observation(self, obs):
        return obs

    def action(self, act):
        return act

    def reward(self):
        return [0.]

    def terminate(self):
        terminate = False
        suc = True
        done = terminate or suc
        info = {
            'success': suc,
            'value_mask': not terminate,
            'reward_mask': not done,
        }
        return done, info

    @property
    def reward_name(self):
        return ['null']

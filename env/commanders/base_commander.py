# -*- coding: utf-8 -*-
import numpy as np


class CommandType:
    DEFAULT = 0
    VELOCITY_FIRST = 1
    HEADING_FIRST = 2


class BaseCommander:
    MAX_FORWARD_ACCELERATION = 2.  # m/s^2
    MAX_LATERAL_ACCELERATION = 1.5  # m/s^2
    MAX_YAW_RATE = 1.5  # rad/s

    def __init__(self, env):
        self.env = env
        self.reset()

    def reset(self):
        self.target_rolling_rate, self.target_forward_velocity, self.target_lateral_velocity, self.target_body_height, self.target_leg_width, self.target_heading = 0., 0., 0., 0., 0., 0.
        self.rolling_rate, self.forward_velocity, self.lateral_velocity, self.body_height, self.leg_width, self.yaw_rate = 0., 0., 0., 0., 0., 0.
        self.velocity = np.copy(self.env.aliengo.base_velocity)
        self.init_heading = self.env.aliengo.rpy[2]
        self._is_randomize = True
        self._enabled = True

    @property
    def command(self):
        return np.asarray([self.rolling_rate, self.forward_velocity, self.lateral_velocity, self.body_height, self.leg_width, self.yaw_rate])

    def enable(self, mode=True):
        self._enabled = mode

    @property
    def name(self):
        return ('rolling_rate', 'forward velocity', 'lateral velocity', 'body height', 'leg width', 'yaw rate')

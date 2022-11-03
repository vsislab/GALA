# -*- coding: utf-8 -*-
"""This file implements an accurate motor model."""

import numpy as np


class Motor:
    """The accurate motor model, which is based on the physics of DC motors."""

    def __init__(self, kp=1.2, kd=0, strength_ratio=1., damping=0.01, dry_friction=0.2):
        self._kp = kp
        self._kd = kd
        self.strength_ratio = strength_ratio
        self.damping = damping
        self.dry_friction = dry_friction

    def convert_to_torque(self, motor_command, motor_position, motor_velocity):
        torque = self._kp * (motor_command - motor_position) - self._kd * motor_velocity
        torque = torque - self.damping * motor_velocity - self.dry_friction * np.sign(motor_velocity)
        return self.strength_ratio * torque
        # print(self.damping * motor_velocity + self.dry_friction * np.sign(motor_velocity))

    def set_KP(self, kp):
        self._kp = kp

    def set_KD(self, kd):
        self._kd = kd

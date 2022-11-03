# -*- coding: utf-8 -*-
import select
from math import pi
import numpy as np
from typing import Tuple

import env.tasks
from env.utils import smallest_signed_angle_between
from .base_commander import BaseCommander
from math import cos, sin
import pybullet as p


class Commander(BaseCommander):
    def __init__(self,
                 env,
                 mode: int,
                 command_duration_time: float,
                 rolling_rate_range: Tuple[float, float],
                 forward_velocity_range: Tuple[float, float],
                 lateral_velocity_range: Tuple[float, float],
                 body_height_range: Tuple[float, float],
                 leg_width_ranged: Tuple[float, float],
                 heading_range: Tuple[float, float]):
        self.command_duration_time = command_duration_time
        self.command_duration_step = command_duration_time / env.control_time_step
        self.rolling_rate_range = rolling_rate_range
        self.forward_velocity_range = forward_velocity_range
        self.lateral_velocity_range = lateral_velocity_range
        self.body_height_range = body_height_range
        self.leg_width_ranged = leg_width_ranged
        self.heading_range = heading_range
        self.terrain_type = env.terrain.param.type.name if env.terrain_randomizer else 'Flat'
        self.count = 0
        self.target_heading = 0
        self.index = 0
        self.mode = mode,
        super(Commander, self).__init__(env)

    def randomize(self):
        self.target_rolling_rate, self.target_forward_velocity, self.target_lateral_velocity, \
        self.target_body_height, self.target_leg_width, self.target_heading_rate = 0., 0., 0., 0, 0., 0.

        if self.heading_range[1] > 0:
            self.target_heading_rate = np.random.uniform(*self.heading_range)

        if self.rolling_rate_range[1] > 0:
            self.target_rolling_rate = np.random.choice([1, -1]) * 1.

        if self.forward_velocity_range[1] > 0:
            if self.forward_velocity_range[1] < 1.5:
                self.target_forward_velocity = np.random.choice([1, -1]) * np.random.uniform(
                    *self.forward_velocity_range)
            else:
                if np.random.uniform() < 0.35:
                    self.target_forward_velocity = np.random.uniform(-1, -2)
                else:
                    self.target_forward_velocity = np.random.uniform(*self.forward_velocity_range)

        if self.lateral_velocity_range[1] > 0:
            self.target_lateral_velocity = np.random.choice([1, -1]) * np.random.uniform(*self.lateral_velocity_range)

        if self.body_height_range[1] > 0:
            self.target_body_height = 5 * np.random.uniform(*self.body_height_range)
            # self.target_forward_velocity = np.random.uniform(*self.forward_velocity_range)

        if self.leg_width_ranged[1] > 0:
            self.target_leg_width = 10 * np.random.uniform(*self.leg_width_ranged)
            # self.target_forward_velocity = 0.8#np.random.uniform(*self.forward_velocity_range)

    def getMatrixFromEuler(self, rpy):
        """each row is the representation of old axis in new axis"""
        quaternion = p.getQuaternionFromEuler(rpy)
        matrix = p.getMatrixFromQuaternion(quaternion)
        return np.reshape(matrix, (3, 3), order='C')

    def _convert_world_to_base_frame(self, quantity, rpy):
        return np.dot(quantity, self.getMatrixFromEuler(rpy))

    def user_debug_text(self):
        c_grey = [0.5, 0.5, 0.5]
        c_red = [1, 0.4, 0.4]
        pos = self.env.aliengo.position + np.array([0., 0, 0.3])
        revovery = False
        textData = "Recovery           "
        if (abs(self.env.aliengo.rpy[0]) > 0.5 or abs(self.env.aliengo.rpy[1]) > 0.5 or
            (self.body_height > 0 and self.env.aliengo.position[2] < 0.25)) and abs(self.rolling_rate) < 1e-2:
            self.env.client.addUserDebugText(text=textData, textPosition=pos + [0, 0, 0.74], textColorRGB=c_red, textSize=2, lifeTime=0.008)
            revovery = True
        else:
            self.env.client.addUserDebugText(text=textData, textPosition=pos + [0, 0, 0.74], textColorRGB=c_grey, textSize=2, lifeTime=0.008)

        textData = "Rolling rate:      " + str(7. * np.round(self.rolling_rate, 2)) + "  rad/s"
        if abs(self.rolling_rate) > 0 and not revovery:
            self.env.client.addUserDebugText(text=textData, textPosition=pos + [0, 0, 0.66], textColorRGB=c_red, textSize=2, lifeTime=0.008)
        else:
            self.env.client.addUserDebugText(text=textData, textPosition=pos + [0, 0, 0.66], textColorRGB=c_grey, textSize=2, lifeTime=0.008)

        textData = "Forward velocity:  " + str(np.round(self.forward_velocity, 2)) + "  m/s"
        if abs(self.forward_velocity) > 0 and not revovery:
            self.env.client.addUserDebugText(text=textData, textPosition=pos + [0, 0, 0.58], textColorRGB=c_red, textSize=2, lifeTime=0.008)
        else:
            self.env.client.addUserDebugText(text=textData, textPosition=pos + [0, 0, 0.58], textColorRGB=c_grey, textSize=2, lifeTime=0.008)

        textData2 = "Lateral velocity: " + str(np.round(self.lateral_velocity, 2)) + "  m/s"
        if abs(self.lateral_velocity) > 0 and not revovery:
            self.env.client.addUserDebugText(text=textData2, textPosition=pos + [0, 0, 0.49], textColorRGB=c_red, textSize=2, lifeTime=0.008)
        else:
            self.env.client.addUserDebugText(text=textData2, textPosition=pos + [0, 0, 0.49], textColorRGB=c_grey, textSize=2, lifeTime=0.008)

        textData3 = "Base height:      " + str(np.round(100. * self.body_height / 5., 2)) + "  cm"
        if abs(self.body_height) > 0 and not revovery:
            self.env.client.addUserDebugText(text=textData3, textPosition=pos + [0, 0, 0.4], textColorRGB=c_red, textSize=2, lifeTime=0.008)
        else:
            self.env.client.addUserDebugText(text=textData3, textPosition=pos + [0, 0, 0.4], textColorRGB=c_grey, textSize=2, lifeTime=0.008)

        textData3 = "Foot width:       " + str(np.round(10. * self.leg_width, 2)) + "  cm"
        if abs(self.leg_width) > 0 and not revovery:
            self.env.client.addUserDebugText(text=textData3, textPosition=pos + [0, 0, 0.3], textColorRGB=c_red, textSize=2, lifeTime=0.008)
        else:
            self.env.client.addUserDebugText(text=textData3, textPosition=pos + [0, 0, 0.3], textColorRGB=c_grey, textSize=2, lifeTime=0.008)

        textData = "Yaw rate:          " + str(np.round(self.yaw_rate, 2)) + "  rad/s"
        if abs(self.yaw_rate) > 0 and not revovery:
            self.env.client.addUserDebugText(text=textData, textPosition=pos + [0, 0, 0.2], textColorRGB=c_red, textSize=2, lifeTime=0.008)
        else:
            self.env.client.addUserDebugText(text=textData, textPosition=pos + [0, 0, 0.2], textColorRGB=c_grey, textSize=2, lifeTime=0.008)

    def train_command(self):
        if self._is_randomize and self.env.counter % self.command_duration_step == 0:
            self.randomize()
        self.yaw_rate = self.target_heading_rate
        self.rolling_rate = self.target_rolling_rate
        self.forward_velocity = self.target_forward_velocity
        self.lateral_velocity = self.target_lateral_velocity
        self.body_height = self.target_body_height
        self.leg_width = self.target_leg_width

    def integrate_command(self):
        self.target_rolling_rate, self.target_forward_velocity, self.target_lateral_velocity, \
        self.target_body_height, self.target_leg_width, self.target_heading_rate = 0., 0., 0., 0, 0., 0.
        self.count += 1
        self.target_forward_velocity = 0.8
        k = 5.
        if self.count < 480:
            self.target_forward_velocity = 0.8
            self.target_body_height = 5 * -0.1
            self.target_lateral_velocity = 0.08
            self.target_lateral_velocity = 8. * (6.5 - self.env.aliengo.position[1])
        elif self.count < 570:
            self.target_forward_velocity = 0.7
            self.target_leg_width = 10 * -0.1
            self.target_lateral_velocity = 8. * (6.5 - self.env.aliengo.position[1])
        elif self.count < 860:
            self.target_forward_velocity = 0.7
            self.target_leg_width = 10 * -0.13
            self.target_lateral_velocity = 0.092
            self.target_lateral_velocity = 8. * (6.5 - self.env.aliengo.position[1])
        elif self.count < 1335:
            self.target_forward_velocity = 0.7
            self.target_leg_width = 10 * 0.13
            self.target_lateral_velocity = -0.07
            self.target_lateral_velocity = 8. * (6.45 - self.env.aliengo.position[1])
        elif self.count < 1590:
            self.target_body_height = 5 * -0.1
        elif self.count < 1785:
            self.target_heading = pi / 2 - 0.05
            k = 1.5
        elif self.count < 1950:
            self.target_heading = pi - 0.05
            k = 1.5
        elif self.count < 2520:
            self.target_heading = pi
            self.target_body_height = 5 * -0.15
            self.target_lateral_velocity = 0.08
            self.target_lateral_velocity = -10. * (8.65 - self.env.aliengo.position[1])
            k = 1.5
        elif self.count < 2960:
            self.target_body_height = 5 * 0.14
            self.target_lateral_velocity = -0.12
            self.target_lateral_velocity = -10. * (8.65 - self.env.aliengo.position[1])
            k = 3.
        elif self.count < 3040:
            self.target_body_height = 0
            self.target_forward_velocity = 3
            k = 1.5
        else:
            self.target_body_height = 0
            self.target_forward_velocity = 0
            k = 0
        if self.count < 3040:
            delta_heading = smallest_signed_angle_between(self.env.aliengo.rpy[2], self.target_heading)
            self.target_heading_rate = np.clip(k * delta_heading, a_min=-self.MAX_YAW_RATE, a_max=self.MAX_YAW_RATE)
        else:
            self.target_heading_rate = 0

    def random_command(self):
        if self._is_randomize and self.env.counter % 500 == 0:
            self.target_rolling_rate, self.target_forward_velocity, self.target_lateral_velocity, \
            self.target_body_height, self.target_leg_width, self.target_heading_rate = 0., 0., 0., 0, 0., 0.
            self.target_heading_rate = np.random.uniform(-1, 1)
            p = np.random.uniform()
            if p < 1 / 6:
                self.target_rolling_rate = np.random.choice([-1, 1]) * 1.  # rolling
                self.target_heading_rate = 0.
            elif p < 2 / 6:
                self.target_forward_velocity = np.random.choice([1, -1]) * np.random.uniform(0.3, 1.3)  # low speed
            elif p < 3 / 6:
                self.target_forward_velocity = np.random.choice([1, -1]) * np.random.uniform(1.3, 3)  # high speed
            elif p < 4 / 6:
                self.target_lateral_velocity = np.random.choice([1, -1]) * np.random.uniform(0.3, 1)  # lateral
            elif p < 5 / 6:
                self.target_forward_velocity = np.random.choice([1, -1]) * np.random.uniform(0.5, 1.)  # height
                self.target_body_height = 5 * np.random.uniform(-0.2, 0.15)
            else:
                self.target_forward_velocity = np.random.choice([1, -1]) * np.random.uniform(0.5, 1.)  # width
                self.target_leg_width = 10 * np.random.uniform(-0.12, 0.12)

    def collect_command(self):
        if self.env.counter % 300 == 0:
            target_rolling_rate_list = [0, -1, -1, -1, 1, 1, 1]
            target_low_forward_velocity_list = [0, 1.2, 0.9, 0.5, -0.5, -0.9, -1.2]
            target_low_forward_velocity_list1 = [0, -1, -0.7, -0.4, 0.4, 0.7, 1]
            target_high_forward_velocity_list = [0, -2, -1.5, -1, 1.5, 2.2, 3]
            target_lateral_velocity_list = [0, -1, -0.6, -0.2, 0.2, 0.6, 1]
            target_body_height_list = [0, -0.2, -0.15, -0.1, 0.05, 0.1, 0.15]
            target_leg_width_list = [0, -0.12, -0.07, -0.03, 0.03, 0.07, 0.12]
            self.target_rolling_rate, self.target_forward_velocity, self.target_lateral_velocity, \
            self.target_body_height, self.target_leg_width, self.target_heading_rate = 0., 0., 0., 0, 0., 0.
            if self.rolling_rate_range[1] > 0:
                self.target_rolling_rate = target_rolling_rate_list[self.index]

            if self.forward_velocity_range[1] > 0:
                if self.forward_velocity_range[1] < 1.5:
                    self.target_forward_velocity = target_low_forward_velocity_list[self.index]
                else:
                    self.target_forward_velocity = target_high_forward_velocity_list[self.index]

            if self.lateral_velocity_range[1] > 0:
                self.target_lateral_velocity = target_lateral_velocity_list[self.index]

            if self.body_height_range[1] > 0:
                self.target_body_height = 5 * target_body_height_list[self.index]
                self.target_forward_velocity = target_low_forward_velocity_list1[self.index]

            if self.leg_width_ranged[1] > 0:
                self.target_leg_width = 10 * target_leg_width_list[self.index]
                self.target_forward_velocity = target_low_forward_velocity_list1[self.index]
            if self.index < len(target_rolling_rate_list) - 1:
                self.index += 1
        if self.heading_range[1] > 0:
            delta_heading = smallest_signed_angle_between(self.env.aliengo.rpy[2], 0)
            self.target_heading_rate = np.clip(1.3 * delta_heading, a_min=-self.MAX_YAW_RATE, a_max=self.MAX_YAW_RATE)

    def smooth_command(self):

        max_forward_velocity_change = self.MAX_FORWARD_ACCELERATION * self.env.control_time_step
        max_lateral_velocity_change = self.MAX_LATERAL_ACCELERATION * self.env.control_time_step
        max_yaw_rate_change = self.MAX_YAW_RATE * self.env.control_time_step

        self.rolling_rate = self.target_rolling_rate

        self.forward_velocity += np.clip(self.target_forward_velocity - self.forward_velocity,
                                         a_min=-max_forward_velocity_change,
                                         a_max=max_forward_velocity_change)

        self.lateral_velocity += np.clip(self.target_lateral_velocity - self.lateral_velocity,
                                         a_min=-max_lateral_velocity_change,
                                         a_max=max_lateral_velocity_change)
        self.leg_width += np.clip(self.target_leg_width - self.leg_width,
                                  a_min=-max_forward_velocity_change,
                                  a_max=max_forward_velocity_change)

        self.body_height += np.clip(self.target_body_height - self.body_height,
                                    a_min=-max_forward_velocity_change,
                                    a_max=max_forward_velocity_change)

        if self.heading_range[1] > 0:
            self.yaw_rate += np.clip(self.target_heading_rate - self.yaw_rate,
                                     a_min=-max_yaw_rate_change,
                                     a_max=max_yaw_rate_change)

    def refresh(self):
        if self.env.mode[0] == 1:
            self.integrate_command()
        elif self.env.mode[0] == 2:
            self.random_command()
            self.user_debug_text()
        else:
            self.train_command()
            self.user_debug_text()
        self.smooth_command()

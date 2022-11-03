# -*- coding: utf-8 -*-
import collections
from math import sin, cos, pi, exp, asin, acos
import numpy as np
from numpy.linalg import norm

from env.utils import IK, PhaseModulator
from env.utils import getMatrixFromEuler
from env.commanders import Commander
from .base_task import BaseTask
from .common import register


def _convert_world_to_base_frame(quantity, rpy):
    return np.dot(quantity, getMatrixFromEuler(rpy))


class LocomotionTask(BaseTask):
    FOOT_POSITION_LOW = np.array([0, -0.2, -0.34])  # defined on the base frame
    FOOT_POSITION_HIGH = np.array([0.15, 0.2, -0.1])

    MOTOR_POSITION_HIGH = np.array([1.222, pi, -0.646]).repeat(4)
    MOTOR_POSITION_LOW = np.array([-1.222, -pi, -2.775]).repeat(4)

    config = {'base_frequency': 0., 'incremental': True, 'motor_position': True}
    if config['motor_position']:
        if config['incremental']:
            config['action_high'] = np.array([1.5, 40, 40, 40]).repeat(4)  # incremental:(f, Vx, Vy, Vz)
            config['action_low'] = -config['action_high']
        else:
            config['action_high'] = np.array([1.5, 0.3, 0.6, 0.6]).repeat(4)  # residual:(f, Vx, Vy, Vz)
            config['action_low'] = -config['action_high']
    else:
        if config['incremental']:
            config['action_high'] = np.array([1.5, 2, 3, 3]).repeat(4)  # incremental:(f, Vx, Vy, Vz)
            config['action_low'] = -config['action_high']
        else:
            config['action_high'] = np.array([1.5, 0.1, 0.15, 0.12]).repeat(4)  # residual:(f, Vx, Vy, Vz)
            config['action_low'] = np.array([-1.5, -0.1, -0.15, 0.0]).repeat(4)
    config['command_duration_time'] = 5.
    config['forward_velocity_range'] = (0.2, 0.6)
    config['lateral_velocity_range'] = (0.2, 0.6)
    config['body_height_range'] = (0.25, 0.45)
    config['leg_width_ranged'] = (0.03, 0.1)
    config['heading_range'] = (-pi, pi)

    def __init__(self, env, **kwargs):
        self.ik = IK(env.aliengo.leg_length)
        self.pms = [PhaseModulator(time_step=env.control_time_step, f0=self.config['base_frequency']) for _ in range(4)]
        self.commander = Commander(env,
                                   self.config['command_duration_time'],
                                   self.config['forward_velocity_range'],
                                   self.config['lateral_velocity_range'],
                                   self.config['body_height_range'],
                                   self.config['leg_width_ranged'],
                                   self.config['heading_range'])
        self.foot_position_reference = env.aliengo.FOOT_POSITION_REFERENCE
        self.motor_position_reference = np.repeat(env.aliengo.STAND_MOTOR_POSITION_REFERENCE, 4)
        self.motor_position_norm = np.repeat([0.5, pi / 4, pi / 2], 4)

        self.action_history = collections.deque(maxlen=3)
        self.net_out_history = collections.deque(maxlen=2)
        self.motor_position = None
        self.ground_impact_force = None
        self.vy_sum = 0
        super(LocomotionTask, self).__init__(env)

    def reset(self):
        self.commander.reset()
        self.vy_sum = 0
        pms_phi0 = np.random.uniform(low=-pi, high=pi, size=4)
        for pm, phi0 in zip(self.pms, pms_phi0):
            pm.reset(phi0=phi0)
        self.pm_frequency = np.zeros(4)
        self.pm_phase = np.zeros(4)
        self.current_foot_position = np.stack([self.foot_position_reference] * 4)
        self.current_action = self.env.aliengo.motor_position
        for _ in range(self.action_history.maxlen):
            self.action_history.append(self.current_action)
        for _ in range(self.net_out_history.maxlen):
            self.net_out_history.append(np.zeros(self.action_space.shape))
        self.refresh(reset=True)
        if self.debug:
            self._debug_param = {
                'command': self.commander.command,
                'net_out': np.zeros(self.action_space.shape),
                'PM_phase': np.asarray([pm.phi for pm in self.pms]),
            }

    def refresh(self, reset=False):
        self.body_contact_state = self.env.get_body_contact_state()
        self.foot_contact_state = self.env.get_foot_contact_state()
        self.foot_scanned_height = self.env.get_scanned_height_around_foot()
        terrain_height = np.mean([self.env.terrain.get_height(self.env.aliengo.position[:2] + np.array(offset))
                                  for offset in [(0., 0.), (0.1, 0), (0, 0.1), (-0.1, 0), (0, -0.1)]])
        self.base_height = self.env.aliengo.position[2]  # - terrain_height
        self.last_ground_impact_force, self.ground_impact_force = self.ground_impact_force, abs(
            self.foot_contact_state['force'])
        self.last_motor_position, self.motor_position = self.motor_position, self.env.aliengo.motor_position
        if reset:
            self.last_motor_position = self.motor_position
            self.last_ground_impact_force = self.ground_impact_force

    def observation(self, obs):
        self.commander.refresh()
        command = [self.commander.forward_velocity, self.commander.body_height, self.commander.yaw_rate]
        # command = [self.commander.forward_velocity, self.commander.yaw_rate]
        base_velocity = _convert_world_to_base_frame(obs['velocity'], obs['rpy'])
        base_rpy_rate = _convert_world_to_base_frame(obs['rpy_rate'], obs['rpy'])
        motor_position = (obs['motor_position'] - self.motor_position_reference) / self.motor_position_norm
        motor_error = obs['motor_position'] - self.current_action
        pm_phase = np.concatenate([[sin(pm.phi), cos(pm.phi)] for pm in self.pms])
        return np.concatenate([
            command,
            base_velocity,
            getMatrixFromEuler(obs['rpy'])[2],
            base_rpy_rate / 2.,
            motor_position,
            obs['motor_velocity'] / 10.,
            motor_error,
            pm_phase,
            self.pm_frequency / 5.,
        ])

    def action(self, net_out):
        self.net_out_history.append(net_out)
        net_out = self.transform(net_out)
        for i in range(4):
            self.pms[i].compute(net_out[i])
        self.pm_frequency = net_out[:4]
        if self.config['motor_position']:
            if self.config['incremental']:
                act = self.current_action + net_out[4:] * self.env.control_time_step  # incremental
            else:
                act = self.motor_position_reference + net_out[4:]  # residual
        else:
            net_out_mat = net_out.reshape((4, 4)).transpose()
            if self.config['incremental']:
                pos = self.current_foot_position + net_out_mat[:, 1:] * self.env.control_time_step  # incremental
            else:
                pos = self.foot_position_reference + net_out_mat[:, 1:]  # residual
            pos = np.clip(pos, self.FOOT_POSITION_LOW, self.FOOT_POSITION_HIGH)
            act = np.stack([self.ik.inverse_kinematics(*p) for p in pos]).transpose().flatten()
            self.current_foot_position = pos
        act = np.clip(act, self.MOTOR_POSITION_LOW, self.MOTOR_POSITION_HIGH)
        self.current_action = act
        self.action_history.append(act)
        if self.debug:
            self._debug_param.update({
                'command': self.commander.command,
                'net_out': net_out,
                'PM_phase': np.asarray([pm.phi for pm in self.pms]),
            })
        return act

    def success(self):
        return False

    def terminate(self):
        """If the aliengo base becomes unstable (based on orientation), the episode terminates early."""
        rpy = self.env.aliengo.rpy
        terminate = self.body_contact_state['num'] > 0 or abs(rpy[0]) > 1 or abs(rpy[1]) > 1
        suc = self.success()
        done = terminate  # or suc
        info = {
            'success': suc,
            'value_mask': not terminate,
            'reward_mask': not done,
        }
        return done, info

    def get_reset_state(self):
        reset_state = {}
        if self.env.terrain_randomizer:
            if self.env.terrain.param.type.name in ['Slope', 'Stair']:
                reset_state['rpy'] = np.asarray([0., 0., np.random.choice([0, pi])])
            else:
                reset_state['rpy'] = np.asarray([0., 0., np.random.uniform(-pi, pi)])
        return reset_state

    def reward(self):
        pass

    reward_name = [
        'x_vel', 'yaw_rate', 'y_vel', 'rp_rate', 'z_vel',
        'foot_support',
        'foot_clear',
        'foot_phase',
        'base_rp',
        'foot_slip',
        'foot_vz',
        'motor_torque',
        'motor_velocity',
        'net_out_smooth',
        'action_smooth',
        'motor_constrain',
        'ground_impact',
        'work',
        'pmf',
        'collision'
    ]

    @property
    def debug_name(self):
        return {
            'command': [*self.commander.name],
            'net_out': [f'{l}_{o}' for o in ['f', 'x', 'y', 'z'] for l in self.env.aliengo.leg_names],
            'PM_phase': [str(l) for l in self.env.aliengo.leg_names],
        }


@register
class ForwardLocomotionTask(LocomotionTask):
    def reward(self):
        command_vel_x_norm = np.clip(abs(self.commander.forward_velocity), a_min=0.3, a_max=None)
        command_yaw_rate_norm = np.clip(abs(self.commander.yaw_rate), a_min=0.1, a_max=None)
        x_vel_rew = exp(-max(min(2 / command_vel_x_norm, 9), 0.7) * (
                self.commander.forward_velocity - self.env.aliengo.base_velocity[0]) ** 2)
        y_vel_rew = exp(-max(min(2 / command_vel_x_norm, 5), 1) * (self.env.aliengo.base_velocity[1]) ** 2)
        foot_slip_rew, foot_support_rew, foot_clear_rew, pmf_rew, motor_constrain_rew = 0., 0., 0., 0., 0.
        motor_position_offset = self.motor_position - self.motor_position_reference
        motor_constrain_rew = -max(min(0.3 / command_vel_x_norm, 1), 0.35) * exp(-0.4 * abs(self.commander.yaw_rate)) * \
                              (3 * norm(motor_position_offset[:4]) ** 2 + norm(motor_position_offset[4:]) ** 2)
        yaw_rate_rew = exp(
            -min(1.6 / command_yaw_rate_norm, 8) * (self.commander.yaw_rate - self.env.aliengo.base_rpy_rate[2]) ** 2)
        rp_rate_rew = exp(-min(2. / command_vel_x_norm, 6) * norm(self.env.aliengo.base_rpy_rate[:2]) ** 2)
        z_vel_rew = exp(-min(3 / command_vel_x_norm, 7) * abs(
            self.env.aliengo.velocity[2] * max(min(0.7 / command_vel_x_norm, 1.5), 1)) ** 2)
        base_rp_rew = -np.clip(norm(self.env.aliengo.rpy[:2]) ** 2, a_min=None, a_max=1)

        foot_support_mask = np.array([0 <= pm.phi < pi for pm in self.pms], dtype=bool)
        foot_swing_mask = np.logical_not(foot_support_mask)
        self.refresh_observation_noise(foot_support_mask)
        scanned_height = np.min(self.foot_scanned_height[foot_swing_mask], axis=1)
        if any(foot_support_mask):
            # foot_support_rew = sum(self.foot_scanned_height.transpose()[0][foot_support_mask] <= 0.01) / sum(
            #     foot_support_mask)
            foot_support_rew = sum(self.foot_contact_state['mask'][foot_support_mask]) / sum(foot_support_mask)
        # foot_support_rew += sum(foot_support_mask) / 4.
        if any(foot_swing_mask):
            foot_clear_rew = (0.6 * sum(scanned_height > 0.01) + 0.6 * sum(scanned_height >= 0.04) - sum(
                scanned_height >= 0.09)) / sum(foot_swing_mask)
        # if any(foot_swing_mask):
        #     foot_clear_rew = sum(self.foot_contact_state['mask'][foot_swing_mask]) / sum(foot_swing_mask)
        if any(self.foot_contact_state['mask']):
            foot_slip_rew = -min(0.1 / command_vel_x_norm, 0.4) * sum(
                (norm(self.env.aliengo.foot_velocity[self.foot_contact_state['mask'], :2], axis=-1) ** 2).clip(max=6))
        foot_phase_match_rew = np.mean(1. - np.logical_xor(foot_support_mask, self.foot_contact_state['mask']))
        # foot_vz_rew = -min(max(0.1 / command_vel_x_norm, 0.06), 0.5) * (norm(self.env.aliengo.foot_velocity[:, 2]) ** 2)
        foot_vz_rew = -0.01 * min(max(0.1 / command_vel_x_norm, 0.02), 0.3) * (
                norm(self.env.aliengo.foot_velocity[:, 2] / (np.min(self.foot_scanned_height, axis=1)).clip(max=0.1, min=0.03)) ** 2)
        motor_torque_rew = -min(0.15 / command_vel_x_norm, 0.2) * norm(self.env.aliengo.motor_torque, ord=1)
        motor_velocity_rew = - max(0.1 / command_vel_x_norm, 0.01) * norm(self.env.aliengo.motor_velocity) ** 2
        net_out_smooth_rew = -min(0.15 / command_vel_x_norm, 0.15) * (
                8 * norm((self.net_out_history[0] - self.net_out_history[1])[:4]) +
                norm((self.net_out_history[0] - self.net_out_history[1])[4:]))
        action_history = np.array(self.action_history)
        action_smooth_rew = -max(min(0.3 / abs(command_vel_x_norm + 0.1), 1), 0.3) * norm(
            action_history[-1] - 2 * action_history[-2] + action_history[-3])
        ground_impact_rew = -min(max(0.1 / command_vel_x_norm, 0.02), 0.2) * norm(
            self.ground_impact_force - self.last_ground_impact_force, ord=1)
        work_rew = -min(0.15 / command_vel_x_norm, 1) * np.abs(
            self.env.aliengo.motor_torque * (self.motor_position - self.last_motor_position)).sum()
        collision_rew = -0.01 * norm(self.body_contact_state['force'], ord=2)
        pmf_rew = -min(0.1 / command_vel_x_norm, 0.3) * sum(self.pm_frequency) / 4.

        rewards = np.array([
            0.5,
            yaw_rate_rew * 1.6,
            x_vel_rew * 2.5,
            y_vel_rew * 1.5,
            rp_rate_rew * 1.5,
            z_vel_rew * 1.5,
            foot_support_rew * 1.,
            foot_clear_rew * 0.7,
            foot_phase_match_rew * 0.5,
            base_rp_rew * 5,
            foot_slip_rew * 0.6,
            foot_vz_rew * 0.1,
            motor_torque_rew * 0.,
            motor_velocity_rew * 0.01,
            net_out_smooth_rew * 0.05,
            action_smooth_rew * 3,
            motor_constrain_rew * 2.7,
            ground_impact_rew * 0.006,
            work_rew * 0.1,
            pmf_rew * 0.2,
            collision_rew * 0.5,
        ])
        # rewards[3:] *= self.schedule_ratio
        rewards = np.clip(rewards, a_min=-2, a_max=3) / 100.
        return rewards

    reward_name = [
        'const', 'yaw_rate',
        'x_vel', 'y_vel', 'rp_rate', 'z_vel',
        'foot_support',
        'foot_clear',
        'foot_phase_match',
        'base_rp',
        'foot_slip',
        'foot_vz',
        'motor_torque',
        'motor_velocity',
        'net_out_smooth',
        'action_smooth',
        'motor_constrain',
        'ground_impact',
        'work',
        'pmf',
        'collision'
    ]


@register
class HighSpeedLocomotionTask(LocomotionTask):
    def reward(self):
        command_vel_x_norm = np.clip(abs(self.commander.forward_velocity), a_min=0.3, a_max=None)
        command_yaw_rate_norm = np.clip(abs(self.commander.yaw_rate), a_min=0.1, a_max=None)
        x_vel_rew = exp(-max(min(2.5 / command_vel_x_norm, 4), 1.2) * (
                self.commander.forward_velocity - self.env.aliengo.base_velocity[0]) ** 2)
        y_vel_rew = exp(-max(min(2 / command_vel_x_norm, 5), 1.5) * (self.env.aliengo.base_velocity[1]) ** 2)
        foot_slip_rew, foot_support_rew, foot_clear_rew, pmf_rew, motor_constrain_rew = 0., 0., 0., 0., 0.
        motor_position_offset = self.motor_position - self.motor_position_reference
        motor_constrain_rew = -max(min(0.3 / command_vel_x_norm, 1), 0.2) * exp(-0.4 * abs(self.commander.yaw_rate)) * \
                              (3 * norm(motor_position_offset[:4]) ** 2 + norm(motor_position_offset[4:]) ** 2)
        yaw_rate_rew = exp(
            -min(1.4 / command_yaw_rate_norm, 8) * (self.commander.yaw_rate - self.env.aliengo.base_rpy_rate[2]) ** 2)
        rp_rate_rew = exp(-min(1.5 / command_vel_x_norm, 6) * norm(self.env.aliengo.base_rpy_rate[:2]) ** 2)
        z_vel_rew = exp(-min(3. / command_vel_x_norm, 7) * abs(
            self.env.aliengo.velocity[2] * max(min(1 / command_vel_x_norm, 1.), 0.5)) ** 2)
        base_rp_rew = -np.clip(norm(self.env.aliengo.rpy[:2]), a_min=None, a_max=1)

        foot_support_mask = np.array([0 <= pm.phi < pi for pm in self.pms], dtype=bool)
        foot_swing_mask = np.logical_not(foot_support_mask)
        self.refresh_observation_noise(foot_support_mask)
        scanned_height = np.min(self.foot_scanned_height[foot_swing_mask], axis=1)
        if any(foot_support_mask):
            # foot_support_rew = sum(self.foot_scanned_height.transpose()[0][foot_support_mask] <= 0.01) / sum(
            #     foot_support_mask)
            foot_support_rew = sum(self.foot_contact_state['mask'][foot_support_mask]) / sum(foot_support_mask)
        # foot_support_rew += sum(foot_support_mask) / 4.
        if any(foot_swing_mask):
            foot_clear_rew = (0.6 * sum(scanned_height > 0.01) + 0.6 * sum(scanned_height >= 0.04) - sum(
                scanned_height >= 0.09)) / sum(foot_swing_mask)
        # if any(foot_swing_mask):
        #     foot_clear_rew = sum(self.foot_contact_state['mask'][foot_swing_mask]) / sum(foot_swing_mask)
        if any(self.foot_contact_state['mask']):
            foot_slip_rew = -min(0.1 / command_vel_x_norm, 0.4) * sum(
                (norm(self.env.aliengo.foot_velocity[self.foot_contact_state['mask'], :2], axis=-1) ** 2).clip(max=6))
        foot_phase_match_rew = np.mean(1. - np.logical_xor(foot_support_mask, self.foot_contact_state['mask']))
        # foot_vz_rew = -min(max(0.1 / command_vel_x_norm, 0.06), 0.5) * (norm(self.env.aliengo.foot_velocity[:, 2]) ** 2)
        foot_vz_rew = -0.01 * min(max(0.1 / command_vel_x_norm, 0.02), 0.3) * (
                norm(self.env.aliengo.foot_velocity[:, 2] / (np.min(self.foot_scanned_height, axis=1)).clip(max=0.1, min=0.03)) ** 2)
        motor_torque_rew = -min(0.15 / command_vel_x_norm, 0.2) * norm(self.env.aliengo.motor_torque, ord=1)
        motor_velocity_rew = - max(0.1 / command_vel_x_norm, 0.01) * norm(self.env.aliengo.motor_velocity) ** 2
        net_out_smooth_rew = -min(0.15 / command_vel_x_norm, 0.15) * (
                8 * norm((self.net_out_history[0] - self.net_out_history[1])[:4]) +
                norm((self.net_out_history[0] - self.net_out_history[1])[4:]))
        action_history = np.array(self.action_history)
        action_smooth_rew = -max(min(0.3 / abs(command_vel_x_norm + 0.1), 1), 0.3) * norm(
            action_history[-1] - 2 * action_history[-2] + action_history[-3])
        ground_impact_rew = -min(max(0.1 / command_vel_x_norm, 0.02), 0.2) * norm(
            self.ground_impact_force - self.last_ground_impact_force, ord=1)
        work_rew = -min(0.15 / command_vel_x_norm, 1) * np.abs(
            self.env.aliengo.motor_torque * (self.motor_position - self.last_motor_position)).sum()
        collision_rew = -0.01 * norm(self.body_contact_state['force'], ord=2)
        pmf_rew = -min(0.1 / command_vel_x_norm, 0.3) * sum(self.pm_frequency) / 4.

        rewards = np.array([
            0.,
            yaw_rate_rew * 1.6,
            x_vel_rew * 2.5,
            y_vel_rew * 1.5,
            rp_rate_rew * 1.5,
            z_vel_rew * 1.5,
            foot_support_rew * 1.,
            foot_clear_rew * 0.7,
            foot_phase_match_rew * 0.5,
            base_rp_rew * 3,
            foot_slip_rew * 0.6,
            foot_vz_rew * 0.1,
            motor_torque_rew * 0.,
            motor_velocity_rew * 0.006,
            net_out_smooth_rew * 0.02,
            action_smooth_rew * 3,
            motor_constrain_rew * 3,
            ground_impact_rew * 0.003,
            work_rew * 0.1,
            pmf_rew * 0.1,
            collision_rew * 0.5,
        ])
        # rewards[3:] *= self.schedule_ratio
        rewards = np.clip(rewards, a_min=-2, a_max=3) / 100.
        return rewards

    reward_name = [
        'const', 'yaw_rate',
        'x_vel', 'y_vel', 'rp_rate', 'z_vel',
        'foot_support',
        'foot_clear',
        'foot_phase_match',
        'base_rp',
        'foot_slip',
        'foot_vz',
        'motor_torque',
        'motor_velocity',
        'net_out_smooth',
        'action_smooth',
        'motor_constrain',
        'ground_impact',
        'work',
        'pmf',
        'collision'
    ]


@register
class LateralLocomotionTask(LocomotionTask):
    def reward(self):
        command_vel_y_norm = np.clip(abs(self.commander.lateral_velocity), a_min=0.3, a_max=None)
        command_yaw_rate_norm = np.clip(abs(self.commander.yaw_rate), a_min=0.1, a_max=None)
        x_vel_rew = exp(-max(min(2 / command_vel_y_norm, 5), 2) * (self.env.aliengo.base_velocity[0]) ** 2)
        y_vel_rew = exp(-max(min(3 / command_vel_y_norm, 8), 2) * (
                self.commander.lateral_velocity - self.env.aliengo.base_velocity[1]) ** 2)
        foot_slip_rew, foot_support_rew, foot_clear_rew, pmf_rew, motor_constrain_rew = 0., 0., 0., 0., 0.
        motor_position_offset = self.motor_position - self.motor_position_reference
        motor_constrain_rew = -exp(-0.4 * abs(self.commander.yaw_rate)) * (max(min(0.2 / command_vel_y_norm, 1), 0.5) *
                                                                           norm(motor_position_offset[:4]) ** 2 + 1.3 * norm(motor_position_offset[4:]) ** 2)
        yaw_rate_rew = exp(
            -min(1.6 / command_yaw_rate_norm, 10) * (self.commander.yaw_rate - self.env.aliengo.base_rpy_rate[2]) ** 2)
        rp_rate_rew = exp(-min(1.5 / command_vel_y_norm, 6) * norm(self.env.aliengo.base_rpy_rate[:2]) ** 2)
        z_vel_rew = exp(-min(3.5 / command_vel_y_norm, 8) * abs(
            self.env.aliengo.velocity[2] * max(min(0.7 / command_vel_y_norm, 1.5), 1)) ** 2)
        base_rp_rew = -np.clip(norm(self.env.aliengo.rpy[:2]) ** 2, a_min=None, a_max=1)

        foot_support_mask = np.array([0 <= pm.phi < pi for pm in self.pms], dtype=bool)
        foot_swing_mask = np.logical_not(foot_support_mask)
        self.refresh_observation_noise(foot_support_mask)
        scanned_height = np.min(self.foot_scanned_height[foot_swing_mask], axis=1)
        if any(foot_support_mask):
            # foot_support_rew = sum(self.foot_scanned_height.transpose()[0][foot_support_mask] <= 0.01) / sum(
            #     foot_support_mask)
            foot_support_rew = sum(self.foot_contact_state['mask'][foot_support_mask]) / sum(foot_support_mask)
        # foot_support_rew += sum(foot_support_mask) / 4.
        if any(foot_swing_mask):
            foot_clear_rew = (0.6 * sum(scanned_height > 0.01) + 0.6 * sum(scanned_height >= 0.04) - sum(
                scanned_height >= 0.09)) / sum(foot_swing_mask)
        # if any(foot_swing_mask):
        #     foot_clear_rew = sum(self.foot_contact_state['mask'][foot_swing_mask]) / sum(foot_swing_mask)
        if any(self.foot_contact_state['mask']):
            foot_slip_rew = -min(0.1 / command_vel_y_norm, 0.4) * sum(
                (norm(self.env.aliengo.foot_velocity[self.foot_contact_state['mask'], :2], axis=-1) ** 2).clip(max=6))
        foot_phase_match_rew = np.mean(1. - np.logical_xor(foot_support_mask, self.foot_contact_state['mask']))
        # foot_vz_rew = -min(max(0.1 / command_vel_y_norm, 0.1), 0.5) * (norm(self.env.aliengo.foot_velocity[:, 2]) ** 2)
        foot_vz_rew = -0.01 * min(max(0.1 / command_vel_y_norm, 0.05), 0.3) * (
                norm(self.env.aliengo.foot_velocity[:, 2] / np.min(self.foot_scanned_height, axis=1).clip(max=0.1, min=0.03)) ** 2)
        motor_torque_rew = -min(0.15 / command_vel_y_norm, 0.2) * norm(self.env.aliengo.motor_torque, ord=1)
        motor_velocity_rew = - max(0.1 / command_vel_y_norm, 0.05) * norm(self.env.aliengo.motor_velocity) ** 2
        net_out_smooth_rew = -min(0.15 / command_vel_y_norm, 0.15) * (
                8 * norm((self.net_out_history[0] - self.net_out_history[1])[:4]) +
                norm((self.net_out_history[0] - self.net_out_history[1])[4:]))
        ground_impact_rew = -min(max(0.1 / command_vel_y_norm, 0.04), 0.12) * norm(
            self.ground_impact_force - self.last_ground_impact_force, ord=1)
        work_rew = -min(0.15 / command_vel_y_norm, 1) * np.abs(
            self.env.aliengo.motor_torque * (self.motor_position - self.last_motor_position)).sum()
        action_history = np.array(self.action_history)
        action_smooth_rew = -max(min(0.3 / abs(command_vel_y_norm + 0.1), 1), 0.3) * norm(
            action_history[-1] - 2 * action_history[-2] + action_history[-3])
        collision_rew = -0.01 * norm(self.body_contact_state['force'], ord=2)
        pmf_rew = -min(0.1 / abs(command_vel_y_norm + 0.2), 0.3) * sum(self.pm_frequency) / 4.

        # rewards[3:] *= self.schedule_ratio
        rewards = np.array([
            1.,
            yaw_rate_rew * 1.6,
            y_vel_rew * 2.5,
            x_vel_rew * 1.5,
            rp_rate_rew * 1.5,
            z_vel_rew * 1.5,
            foot_support_rew * 1.,
            foot_clear_rew * 0.7,
            foot_phase_match_rew * 0.5,
            base_rp_rew * 3,
            foot_slip_rew * 0.7,
            foot_vz_rew * 0.1,
            motor_torque_rew * 0.,
            motor_velocity_rew * 0.01,
            net_out_smooth_rew * 0.05,
            action_smooth_rew * 3,
            motor_constrain_rew * 2.7,
            ground_impact_rew * 0.006,
            work_rew * 0.1,
            pmf_rew * 0.2,
            collision_rew * 0.5,
        ])
        # rewards[3:] *= self.schedule_ratio
        rewards = np.clip(rewards, a_min=-2, a_max=3) / 100.
        return rewards

    reward_name = [
        'const', 'yaw_rate',
        'y_vel', 'x_vel', 'rp_rate', 'z_vel',
        'foot_support',
        'foot_clear',
        'foot_phase_match',
        'base_rp',
        'foot_slip',
        'foot_vz',
        'motor_torque',
        'motor_velocity',
        'net_out_smooth',
        'action_smooth',
        'motor_constrain',
        'ground_impact',
        'work',
        'pmf',
        'collision'
    ]


@register
class HeightLocomotionTask(LocomotionTask):
    def reward(self):
        self.vy_sum += self.env.aliengo.base_velocity[1]
        command_h_norm = np.clip(abs(self.commander.body_height), a_min=0.3, a_max=None)
        command_vel_x_norm = np.clip(abs(self.commander.forward_velocity), a_min=0.3, a_max=None)
        command_yaw_rate_norm = np.clip(abs(self.commander.yaw_rate), a_min=0.1, a_max=None)
        base_height = 5 * (self.base_height - 0.35)
        h_rew = exp(-min(max(4. / command_h_norm, 3), 8) * (self.commander.body_height - base_height) ** 2)
        x_vel_rew = exp(-max(min(3. / command_vel_x_norm, 8), 3) * (
                self.commander.forward_velocity - self.env.aliengo.base_velocity[0]) ** 2)
        y_vel_rew = exp(-max(min(3 / command_vel_x_norm, 8), 4) * (self.env.aliengo.base_velocity[1]) ** 2)
        foot_slip_rew, foot_support_rew, foot_clear_rew, pmf_rew, motor_constrain_rew = 0., 0., 0., 0., 0.
        theta = min(asin(abs(self.commander.body_height / 5. + 0.35) / 0.5), pi / 2.2)
        motor_position_reference = np.repeat(np.array([0., pi / 2 - theta, 2 * theta - pi]), 4)
        motor_position_offset = self.motor_position - motor_position_reference
        motor_constrain_rew = - max(min(0.3 / command_vel_x_norm, 1), 0.4) * exp(-0.4 * abs(self.commander.yaw_rate)) * \
                              (2.9 * norm(motor_position_offset[:4]) ** 2 + norm(motor_position_offset[4:8]) ** 2 + 0.15 * norm(motor_position_offset[8:]) ** 2)
        yaw_rate_rew = exp(
            -min(1.6 / command_yaw_rate_norm, 10) * (self.commander.yaw_rate - self.env.aliengo.base_rpy_rate[2]) ** 2)
        rp_rate_rew = exp(-min(2. / command_vel_x_norm, 6) * norm(self.env.aliengo.base_rpy_rate[:2]) ** 2)
        z_vel_rew = exp(-max(min(3 / command_vel_x_norm, 7), 3) * abs(
            self.env.aliengo.velocity[2] * max(min(0.7 / command_vel_x_norm, 1.5), 1)) ** 2)
        base_rp_rew = -np.clip(norm(self.env.aliengo.rpy[:2]) ** 2, a_min=None, a_max=1)

        foot_support_mask = np.array([0 <= pm.phi < pi for pm in self.pms], dtype=bool)
        foot_swing_mask = np.logical_not(foot_support_mask)
        self.refresh_observation_noise(foot_support_mask)
        # scanned_height = np.min(self.foot_scanned_height[foot_swing_mask], axis=1)
        scanned_height = self.env.aliengo.foot_position[:, 2] - 0.03
        if any(foot_support_mask):
            # foot_support_rew = sum(self.foot_scanned_height.transpose()[0][foot_support_mask] <= 0.01) / sum(
            #     foot_support_mask)
            foot_support_rew = sum(self.foot_contact_state['mask'][foot_support_mask]) / sum(foot_support_mask)
        # foot_support_rew += sum(foot_support_mask) / 4.
        if any(foot_swing_mask):
            foot_clear_rew = (0.6 * sum(scanned_height[foot_support_mask] > 0.01) + 0.6 * sum(scanned_height[foot_support_mask] >= 0.05) - sum(
                scanned_height[foot_support_mask] >= 0.09)) / sum(foot_swing_mask)
        # if any(foot_swing_mask):
        #     foot_clear_rew = sum(self.foot_contact_state['mask'][foot_swing_mask]) / sum(foot_swing_mask)
        if any(self.foot_contact_state['mask']):
            foot_slip_rew = -min(0.1 / command_vel_x_norm, 0.4) * sum(
                (norm(self.env.aliengo.foot_velocity[self.foot_contact_state['mask'], :2], axis=-1) ** 2).clip(max=6))
        foot_phase_match_rew = np.mean(1. - np.logical_xor(foot_support_mask, self.foot_contact_state['mask']))
        # foot_vz_rew = -min(max(0.1 / command_vel_x_norm, 0.1), 0.5) * (norm(self.env.aliengo.foot_velocity[:, 2]) ** 2)
        foot_vz_rew = -0.01 * min(max(0.1 / command_vel_x_norm, 0.05), 0.3) * (
                norm(self.env.aliengo.foot_velocity[:, 2] / np.min(self.foot_scanned_height, axis=1).clip(max=0.1, min=0.03)) ** 2)
        motor_torque_rew = -min(0.15 / command_vel_x_norm, 0.2) * norm(self.env.aliengo.motor_torque, ord=1)
        motor_velocity_rew = - max(0.1 / command_vel_x_norm, 0.03) * norm(self.env.aliengo.motor_velocity) ** 2
        net_out_smooth_rew = -min(0.15 / command_vel_x_norm, 0.15) * (
                8 * norm((self.net_out_history[0] - self.net_out_history[1])[:4]) +
                norm((self.net_out_history[0] - self.net_out_history[1])[4:]))
        ground_impact_rew = -min(max(0.1 / command_vel_x_norm, 0.04), 0.12) * norm(
            self.ground_impact_force - self.last_ground_impact_force, ord=1)
        vy_sum_rew = -abs(self.vy_sum / 100.) ** 0.5
        work_rew = -min(0.15 / command_vel_x_norm, 1) * np.abs(
            self.env.aliengo.motor_torque * (self.motor_position - self.last_motor_position)).sum()
        action_history = np.array(self.action_history)
        action_smooth_rew = -max(min(0.3 / abs(command_vel_x_norm + 0.1), 1), 0.3) * norm(
            action_history[-1] - 2 * action_history[-2] + action_history[-3])
        collision_rew = -0.01 * norm(self.body_contact_state['force'], ord=2)
        pmf_rew = -min(0.1 / command_vel_x_norm, 0.3) * sum(self.pm_frequency) / 4.

        rewards = np.array([
            0.,
            yaw_rate_rew * 1.6,
            h_rew * 3.,
            x_vel_rew * 2,
            y_vel_rew * 1.6,
            rp_rate_rew * 1.5,
            z_vel_rew * 1.5,
            foot_support_rew * 1.,
            foot_clear_rew * 0.5,
            foot_phase_match_rew * 0.5,
            vy_sum_rew * 1.5,
            base_rp_rew * 3,
            foot_slip_rew * 0.6,
            foot_vz_rew * 0.1,
            motor_torque_rew * 0.,
            motor_velocity_rew * 0.01,
            net_out_smooth_rew * 0.05,
            action_smooth_rew * 3,
            motor_constrain_rew * 2.7,
            ground_impact_rew * 0.006,
            work_rew * 0.1,
            pmf_rew * 0.2,
            collision_rew * 0.5,
        ])
        # rewards[4:] *= self.schedule_ratio
        rewards = np.clip(rewards, a_min=-2, a_max=3) / 100.
        return rewards

    reward_name = [
        'const', 'yaw_rate', 'h',
        'x_vel', 'y_vel', 'rp_rate', 'z_vel',
        'foot_support',
        'foot_clear',
        'foot_phase_match', 'vy_sum',
        'base_rp',
        'foot_slip',
        'foot_vz',
        'motor_torque',
        'motor_velocity',
        'net_out_smooth',
        'action_smooth',
        'motor_constrain',
        'ground_impact',
        'work',
        'pmf',
        'collision'
    ]


@register
class WidthLocomotionTask(LocomotionTask):
    def observation(self, obs):
        self.commander.refresh()
        command = [self.commander.forward_velocity, self.commander.leg_width, self.commander.yaw_rate]
        # command = [self.commander.forward_velocity, self.commander.yaw_rate]
        base_velocity = _convert_world_to_base_frame(obs['velocity'], obs['rpy'])
        base_rpy_rate = _convert_world_to_base_frame(obs['rpy_rate'], obs['rpy'])
        motor_position = (obs['motor_position'] - self.motor_position_reference) / self.motor_position_norm
        motor_error = obs['motor_position'] - self.current_action
        pm_phase = np.concatenate([[sin(pm.phi), cos(pm.phi)] for pm in self.pms])
        return np.concatenate([
            command,
            base_velocity,
            getMatrixFromEuler(obs['rpy'])[2],
            base_rpy_rate / 2.,
            motor_position,
            obs['motor_velocity'] / 10.,
            motor_error,
            pm_phase,
            self.pm_frequency / 5.,
        ])

    def reward(self):
        self.vy_sum += self.env.aliengo.base_velocity[1]
        command_w_norm = np.clip(abs(self.commander.leg_width), a_min=0.2, a_max=None)
        command_vel_x_norm = np.clip(abs(self.commander.forward_velocity), a_min=0.3, a_max=None)
        command_yaw_rate_norm = np.clip(abs(self.commander.yaw_rate), a_min=0.1, a_max=None)
        foot_position_offset = 10 * (self.env.aliengo.get_foot_position_on_base_frame()[:, 0] - 0.08)
        w_rew = sum(np.exp(
            -max(min(4.5 / command_w_norm, 8), 3.) * (self.commander.leg_width - foot_position_offset) ** 2)) / 4.
        x_vel_rew = exp(-max(min(3 / command_vel_x_norm, 7), 3) * (
                self.commander.forward_velocity - self.env.aliengo.base_velocity[0]) ** 2)  # + \
        # min(self.commander.forward_velocity, self.env.aliengo.base_velocity[0])
        # x_vel_rew = min(self.commander.forward_velocity, self.env.aliengo.base_velocity[0])
        y_vel_rew = exp(-max(min(3 / command_vel_x_norm, 9), 5) * (self.env.aliengo.base_velocity[1]) ** 2)
        vy_sum_rew = -abs(self.vy_sum / 100.) ** 0.5
        foot_slip_rew, foot_support_rew, foot_clear_rew, pmf_rew, motor_constrain_rew = 0., 0., 0., 0., 0.
        alpha = asin(self.commander.leg_width / 10. / 0.25 / 1.414)
        motor_position_reference = np.concatenate([np.asarray([-alpha, alpha, -alpha, alpha]), self.motor_position_reference[4:]])
        motor_position_offset = self.motor_position - motor_position_reference  # self.motor_position_reference
        motor_constrain_rew = -max(min(0.3 / command_vel_x_norm, 1), 0.3) * exp(-0.4 * abs(self.commander.yaw_rate)) * \
                              (2.5 * norm(motor_position_offset[:4]) ** 2 + 1.3 * norm(motor_position_offset[4:]) ** 2)
        yaw_rate_rew = exp(
            -min(1.6 / command_yaw_rate_norm, 10) * (self.commander.yaw_rate - self.env.aliengo.base_rpy_rate[2]) ** 2)
        rp_rate_rew = exp(-min(2. / command_vel_x_norm, 4.5) * norm(self.env.aliengo.base_rpy_rate[:2]) ** 2)
        z_vel_rew = exp(-max(min(3 / command_vel_x_norm, 8), 3) * abs(
            self.env.aliengo.velocity[2] * max(min(0.7 / command_vel_x_norm, 1.5), 1)) ** 2)
        base_rp_rew = -np.clip(norm(self.env.aliengo.rpy[:2]) ** 2, a_min=None, a_max=1)
        foot_support_mask = np.array([0 <= pm.phi < pi for pm in self.pms], dtype=bool)
        foot_swing_mask = np.logical_not(foot_support_mask)
        self.refresh_observation_noise(foot_support_mask)
        scanned_height = np.min(self.foot_scanned_height[foot_swing_mask], axis=1)
        if any(foot_support_mask):
            # foot_support_rew = sum(self.foot_scanned_height.transpose()[0][foot_support_mask] <= 0.01) / sum(
            #     foot_support_mask)
            foot_support_rew = sum(self.foot_contact_state['mask'][foot_support_mask]) / sum(foot_support_mask)
        # foot_support_rew += sum(foot_support_mask) / 4.
        if any(foot_swing_mask):
            foot_clear_rew = (0.6 * sum(scanned_height > 0.01) + 0.6 * sum(scanned_height >= 0.04) - sum(
                scanned_height >= 0.09)) / sum(foot_swing_mask)
        # if any(foot_swing_mask):
        #     foot_clear_rew = sum(self.foot_contact_state['mask'][foot_swing_mask]) / sum(foot_swing_mask)
        if any(self.foot_contact_state['mask']):
            foot_slip_rew = -min(0.1 / command_vel_x_norm, 0.4) * sum(
                (norm(self.env.aliengo.foot_velocity[self.foot_contact_state['mask'], :2], axis=-1) ** 2).clip(max=6))
        foot_phase_match_rew = np.mean(1. - np.logical_xor(foot_support_mask, self.foot_contact_state['mask']))
        # foot_vz_rew = -min(max(0.1 / command_vel_x_norm, 0.1), 0.8) * (norm(self.env.aliengo.foot_velocity[:, 2]) ** 2)
        foot_vz_rew = -0.01 * min(max(0.1 / command_vel_x_norm, 0.05), 0.3) * (
                norm(self.env.aliengo.foot_velocity[:, 2] / np.min(self.foot_scanned_height, axis=1).clip(max=0.1, min=0.03)) ** 2)
        motor_torque_rew = -min(0.15 / command_vel_x_norm, 0.2) * norm(self.env.aliengo.motor_torque, ord=1)
        motor_velocity_rew = - max(0.1 / command_vel_x_norm, 0.03) * norm(self.env.aliengo.motor_velocity) ** 2
        net_out_smooth_rew = -min(0.15 / command_vel_x_norm, 0.15) * (
                8 * norm((self.net_out_history[0] - self.net_out_history[1])[:4]) +
                norm((self.net_out_history[0] - self.net_out_history[1])[4:]))
        ground_impact_rew = -min(max(0.1 / command_vel_x_norm, 0.04), 0.12) * norm(
            self.ground_impact_force - self.last_ground_impact_force, ord=1)
        work_rew = -min(0.15 / command_vel_x_norm, 1) * np.abs(
            self.env.aliengo.motor_torque * (self.motor_position - self.last_motor_position)).sum()
        action_history = np.array(self.action_history)
        action_smooth_rew = -max(min(0.3 / abs(command_vel_x_norm + 0.1), 1), 0.3) * norm(
            action_history[-1] - 2 * action_history[-2] + action_history[-3])
        collision_rew = -0.01 * norm(self.body_contact_state['force'], ord=2)
        pmf_rew = -min(0.1 / command_vel_x_norm, 0.3) * sum(self.pm_frequency[foot_swing_mask]) / 4.

        rewards = np.array([
            0,
            yaw_rate_rew * 1.6,
            w_rew * 3,
            x_vel_rew * 2,
            y_vel_rew * 2.5,
            rp_rate_rew * 1.5,
            z_vel_rew * 1.5,
            foot_support_rew * 1.,
            foot_clear_rew * 0.6,
            foot_phase_match_rew * 0.5,
            vy_sum_rew * 1.5,
            base_rp_rew * 2,
            foot_slip_rew * 0.7,
            foot_vz_rew * 0.1,
            motor_torque_rew * 0.,
            motor_velocity_rew * 0.01,
            net_out_smooth_rew * 0.05,
            action_smooth_rew * 3,
            motor_constrain_rew * 3.,
            ground_impact_rew * 0.006,
            work_rew * 0.1,
            pmf_rew * 0.2,
            collision_rew * 0.5,
        ])
        # rewards[5:] *= self.schedule_ratio
        rewards = np.clip(rewards, a_min=-2, a_max=3) / 100.
        return rewards

    reward_name = [
        'const', 'yaw_rate',
        'w',
        'x_vel', 'y_vel', 'rp_rate', 'z_vel',
        'foot_support',
        'foot_clear',
        'foot_phase_match', 'vy_sum',
        'base_rp',
        'foot_slip',
        'foot_vz',
        'motor_torque',
        'motor_velocity',
        'net_out_smooth',
        'action_smooth',
        'motor_constrain',
        'ground_impact',
        'work',
        'pmf',
        'collision'
    ]

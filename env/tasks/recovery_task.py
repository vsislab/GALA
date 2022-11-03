# -*- coding: utf-8 -*-
import collections
from math import sin, cos, pi, exp, asin
import numpy as np
from numpy.linalg import norm

from env.utils import smallest_signed_angle_between, getMatrixFromEuler, pose3d
from .base_task import BaseTask
from .common import register
from env.commanders import Commander


def _convert_world_to_base_frame(quantity, rpy):
    return np.dot(quantity, getMatrixFromEuler(rpy))


@register
class RecoveryTask(BaseTask):
    MOTOR_POSITION_HIGH = np.array([1.222, pi, -0.646]).repeat(4)
    MOTOR_POSITION_LOW = np.array([-1.222, -pi, -2.775]).repeat(4)

    action_high = np.array([10.6, 10.6, 10.6]).repeat(4)  # incremental:(Vx, Vy, Vz)
    config = {
        'action_high': action_high,
        'action_low': -action_high
    }

    def __init__(self, env, **kwargs):
        self.commander = Commander(env,
                                   self.config['command_duration_time'],
                                   self.config['rolling_rate_range'],
                                   self.config['forward_velocity_range'],
                                   self.config['lateral_velocity_range'],
                                   self.config['body_height_range'],
                                   self.config['leg_width_ranged'],
                                   self.config['heading_range'])
        self.motor_position_reference = np.repeat(env.aliengo.STAND_MOTOR_POSITION_REFERENCE, 4)
        self.motor_position_norm = np.repeat([0.5, pi / 4, pi / 2], 4)
        self.pm_frequency = np.zeros(1)
        self.pms = np.zeros(4)
        self.action_history = collections.deque(maxlen=10)
        self.base_flip_history = collections.deque(maxlen=10)
        self.reset_time = 0.
        self.reset_states = []
        super(RecoveryTask, self).__init__(env)

    def reset(self):
        if self.reset_time > 0:
            self.reset_states.append(self.env.aliengo.state_dict)
        self.recover_start_position = self.env.aliengo.position
        self.current_action = self.env.aliengo.motor_position
        self.ground_impact_force = None
        for _ in range(self.action_history.maxlen):
            self.action_history.append(self.current_action)
        self.refresh()
        self.last_ground_impact_force = self.ground_impact_force
        self.last_motor_position = self.motor_position
        if self.debug:
            self._debug_param = {
                'net_out': np.zeros(self.action_space.shape),
            }

    def refresh(self):
        self.self_contact_state = self.env.get_self_contact_state()
        self.body_contact_state = self.env.get_body_contact_state()
        self.foot_contact_state = self.env.get_foot_contact_state()
        self.last_ground_impact_force, self.ground_impact_force = self.ground_impact_force, abs(
            self.foot_contact_state['force'])
        terrain_height = np.mean(
            [self.env.terrain.get_height(self.env.aliengo.position[:2] + np.array(offset)) for offset in
             [(0., 0.), (0.1, 0), (0, 0.1), (-0.1, 0), (0, -0.1)]])
        self.base_height = self.env.aliengo.position[2] - terrain_height
        self.base_flip = np.array([0, 0, 1]).dot(
            pose3d.QuaternionRotatePoint(self.env.aliengo.orientation, np.array([0, 0, 1])))

    def observation(self, obs):
        base_velocity = _convert_world_to_base_frame(obs['velocity'], obs['rpy'])
        base_rpy_rate = _convert_world_to_base_frame(obs['rpy_rate'], obs['rpy'])
        motor_position = (obs['motor_position'] - self.motor_position_reference) / self.motor_position_norm
        motor_error = obs['motor_position'] - self.current_action
        return np.concatenate([
            # [self.base_height / 0.3],
            base_velocity,
            getMatrixFromEuler(obs['rpy'])[2],
            base_rpy_rate / 2.,
            motor_position,
            obs['motor_velocity'] / 10.,
            motor_error,
        ])

    def action(self, net_out):
        net_out = self.transform(net_out)
        act = self.current_action + net_out * self.env.control_time_step  # incremental
        act = np.clip(act, a_min=self.MOTOR_POSITION_LOW, a_max=self.MOTOR_POSITION_HIGH)
        self.current_action = act
        self.action_history.append(act)
        if self.debug:
            self._debug_param.update({
                'net_out': net_out,
            })
        return act

    def terminate(self):
        terminate = norm(self.env.aliengo.position - self.recover_start_position) > 5
        done = terminate  # or self.env.terrain_penetration
        info = {
            'success': False,
            'reward_mask': not done,
            'value_mask': not terminate,
        }
        return done, info

    def get_reset_state(self):
        if self.env.terrain_updated:
            self.reset_states.clear()
        reset_state = {}
        self.reset_time = 0.3
        reset_state['reset_time'] = self.reset_time
        reset_state['reset_motor_position'] = np.repeat([0, 1.4, -2.5], 4)  # fold
        p = np.random.uniform()
        if p <= 0.4:
            reset_state['reset_time'] = self.reset_time
            reset_state['position'] = np.array([-1., 0., 0.4])
            reset_state['motor_position'] = reset_state['reset_motor_position'] + \
                                            np.random.uniform(low=np.array([-0.2, -0.2, -0.1]).repeat(4),
                                                              high=np.array([0.2, 0.2, 0.2]).repeat(4))
        else:
            reset_state['position'] = np.array([-1, 0., 0.4])
            reset_state['rpy'] = np.random.uniform(low=[-pi, -0., -pi], high=[pi, 0., pi])
            reset_state['motor_position'] = reset_state['reset_motor_position'] + \
                                            np.random.uniform(low=np.array([-1, -pi / 6, -0.1]).repeat(4),
                                                              high=np.array([1, pi / 6, 0.7]).repeat(4))
        reset_state['motor_position'] = np.clip(reset_state['motor_position'], a_min=self.MOTOR_POSITION_LOW,
                                                a_max=self.MOTOR_POSITION_HIGH)
        return reset_state

    def reward(self):
        flip_rew = 0.5 * (self.base_flip + 1)
        self_collision_rew = -self.self_contact_state['num']
        motor_torque_rew = -norm(self.env.aliengo.motor_torque, ord=1)
        motor_velocity_rew = -norm(self.env.aliengo.motor_velocity, ord=2)
        action_history = np.array(self.action_history)
        action_smooth_rew = -norm(action_history[-1] - 2 * action_history[-2] + action_history[-3])
        rpy_rate_rew = -norm(self.env.aliengo.base_rpy_rate * np.array([0., 1, 1]), ord=2) ** 2 - 2. * max(abs(self.env.aliengo.base_rpy_rate[0]) - 7, 0) ** 2
        base_vel_rew = -norm(self.env.aliengo.velocity, ord=2).clip(max=6, min=0)
        collision_rew = -0.01 * norm(self.body_contact_state['force'], ord=2) ** 2 / self.body_contact_state['num'] if \
            self.body_contact_state['num'] > 0 else 0.
        foot_slip_rew, base_height_rew, base_rp_rew, foot_vz_rew, foot_contact_num_rew = 0., 0., 0., 0, 0
        support_foot_mask = self.foot_contact_state['mask'].astype(bool)
        self.refresh_observation_noise(support_foot_mask)
        motor_fold_position_offset = np.array([0, 1.4, -2.4]).repeat(4) - self.env.aliengo.motor_position
        motor_constrain_rew = -norm(np.repeat([1.2, 0.7, 0.7], 4) * motor_fold_position_offset, ord=2) ** 2
        ground_impact_rew = -norm(self.ground_impact_force - self.last_ground_impact_force, ord=1)
        foot_contact_force_rew = -norm(self.foot_contact_state['force'] - np.mean(self.foot_contact_state['force']), ord=1)
        foot_slip_rew = - (norm(self.env.aliengo.foot_velocity[:2]) ** 2).clip(max=10)
        foot_vz_rew = - min(norm(self.env.aliengo.foot_velocity) ** 2, 20)
        foot_height_rew = -norm(np.min(self.foot_scanned_height, axis=1), ord=1)
        base_height_rew = - min(self.base_height, 0.25)
        pmf_rew = -0.3 * sum(self.pm_frequency) / 4.
        base_rp_rew = - norm(self.env.aliengo.rpy[:2], ord=1).clip(max=0.3)
        motor_error_rew = -norm(self.motor_error, ord=2)
        if any(support_foot_mask):
            foot_contact_num_rew = flip_rew * sum(support_foot_mask) / 4.
        if self.base_flip >= 0.95 and self.base_flip_history[4] >= 0.95 and \
                np.all(np.abs(self.env.aliengo.motor_position[:4]) < 0.5) and \
                np.all(np.abs(np.abs(self.env.aliengo.motor_position[4:8]) - 1.2) < 0.8) and \
                norm(self.env.aliengo.base_rpy_rate) < 4 and norm(self.env.aliengo.velocity) < 3:
            if any(support_foot_mask):
                foot_contact_num_rew = sum(support_foot_mask) / 4.
            base_vel_rew = - norm(self.env.aliengo.velocity * np.array([1, 1., 3.]), ord=1).clip(max=6)
            base_height_rew = exp(-8.5 * (self.base_height - 0.35)) if self.base_height < 0.35 else 1.1
            motor_stand_position_offset = self.env.aliengo.STAND_MOTOR_POSITION_REFERENCE.repeat(4) - self.env.aliengo.motor_position
            motor_constrain_rew = 10. * exp(-0.25 * norm(np.repeat([1.5, 0.7, 0.7], 4) * motor_stand_position_offset, ord=1) ** 2)
        rewards = np.array([
            5,
            flip_rew * 3.,
            foot_contact_num_rew * 3.,
            base_height_rew * 2.,
            motor_constrain_rew * 0.3,
            foot_height_rew * 0.05,
            base_vel_rew * 0.3,
            rpy_rate_rew * 0.1,
            base_rp_rew * 2,
            foot_slip_rew * 0.02,
            foot_vz_rew * 0.015,
            motor_torque_rew * 0.,
            motor_velocity_rew * 0.02,
            action_smooth_rew * 1.,
            collision_rew * 0.0,
            ground_impact_rew * 0.001,
            self_collision_rew * 1,
            foot_contact_force_rew * 0.00,
            motor_error_rew * 0.1,
            pmf_rew * 0.
        ])
        # rewards[5:] *= self.schedule_ratio
        rewards = np.clip(rewards, a_min=-7., a_max=5) / 100.
        return rewards

    reward_name = [
        'const',
        'flip', 'foot_contact_num',
        'base_height',
        'motor_constrain',
        'foot_height',
        'base_vel',
        'rpy_rate',
        'rp',
        'foot_slip',
        'foot_vz',
        'motor_torque',
        'motor_velocity',
        'action_smooth',
        'collision_force',
        'ground_impact',
        'self_collision',
        'foot_contact_force',
        'motor_error',
        'pmf'
    ]

    @property
    def debug_name(self):
        return {
            'net_out': [f'{l}_{o}' for o in ['x', 'y', 'z'] for l in self.env.aliengo.leg_names],
        }


@register
class RollingTask(RecoveryTask):
    def terminate(self):
        terminate = norm(self.env.aliengo.base_rpy_rate[1:]) > 6 or self.base_height > 1
        done = terminate  # or self.env.terrain_penetration
        info = {
            'success': False,
            'reward_mask': not done,
            'value_mask': not terminate,
        }
        return done, info

    def get_reset_state(self):
        if self.env.terrain_updated:
            self.reset_states.clear()
        reset_state = {}
        self.reset_time = 0.5
        reset_state['reset_time'] = self.reset_time
        reset_state['reset_motor_position'] = np.repeat([0, 1.4, -2.5], 4)  # fold
        p = np.random.uniform()
        if p <= 0.4:
            reset_state['reset_time'] = self.reset_time
            reset_state['position'] = np.array([-1., 0., 0.4])
            reset_state['motor_position'] = reset_state['reset_motor_position'] + \
                                            np.random.uniform(low=np.array([-0.2, -0.2, -0.1]).repeat(4),
                                                              high=np.array([0.2, 0.2, 0.2]).repeat(4))
        else:
            reset_state['position'] = np.array([-1, 0., 0.4])
            reset_state['rpy'] = np.random.uniform(low=[-pi, -0., -pi], high=[pi, 0., pi])
            reset_state['motor_position'] = reset_state['reset_motor_position'] + \
                                            np.random.uniform(low=np.array([-1, -pi / 6, -0.1]).repeat(4),
                                                              high=np.array([1, pi / 6, 0.7]).repeat(4))
        reset_state['motor_position'] = np.clip(reset_state['motor_position'], a_min=self.MOTOR_POSITION_LOW,
                                                a_max=self.MOTOR_POSITION_HIGH)
        return reset_state

    def reward(self):
        foot_support_rew = 0
        support_foot_mask = self.foot_contact_state['mask'].astype(bool)
        if any(support_foot_mask):
            foot_support_rew = sum(support_foot_mask) / 4.
        self.refresh_observation_noise(support_foot_mask)
        r_rate_rew = min(2 * self.commander.rolling_rate * self.env.aliengo.base_rpy_rate[0], 15) - \
                     max(abs(self.env.aliengo.base_rpy_rate[0]) - 8, 0) ** 2
        base_height_rew = -5 * min(self.base_height, 0.25) ** 2
        foot_height_rew = -norm(np.min(self.foot_scanned_height, axis=1)) ** 2
        foot_slip_rew = - (norm(self.env.aliengo.foot_velocity[:2]) ** 2).clip(max=10)
        foot_vz_rew = - min(norm(self.env.aliengo.foot_velocity) ** 2, 30)
        py_rate_rew = -norm(self.env.aliengo.base_rpy_rate[1:], ord=1).clip(max=20)
        base_vel_rew = -self.env.aliengo.base_velocity[0] ** 2 - 2 * self.env.aliengo.velocity[2] ** 2
        motor_fold_position_offset = np.array([0, 1.4, -2.5]).repeat(4) - self.env.aliengo.motor_position
        motor_constrain_rew = -norm(np.repeat([1.2, 0.7, 0.7], 4) * motor_fold_position_offset, ord=2) ** 2
        ground_impact_rew = -norm(self.ground_impact_force - self.last_ground_impact_force, ord=1)
        self_collision_rew = -self.self_contact_state['num']
        motor_torque_rew = -norm(self.env.aliengo.motor_torque, ord=1)
        motor_velocity_rew = -norm(self.env.aliengo.motor_velocity, ord=2)
        action_history = np.array(self.action_history)
        action_smooth_rew = -norm(action_history[-1] - 2 * action_history[-2] + action_history[-3])
        collision_rew = max(
            - 0.01 * norm(self.body_contact_state['force'], ord=2) ** 2 / self.body_contact_state['num'], -3000) if \
            self.body_contact_state['num'] > 0 else 0.
        pmf_rew = -0.2 * sum(self.pm_frequency) / 4.
        motor_error_rew = -norm(self.motor_error, ord=2)

        rewards = np.array([
            3,
            r_rate_rew * 1,
            foot_support_rew * 2.,
            base_height_rew * 1,
            foot_height_rew * 0.8,
            motor_constrain_rew * 0.35,
            base_vel_rew * 0.01,
            py_rate_rew * 0.01,
            foot_slip_rew * 0.01,
            foot_vz_rew * 0.01,
            motor_torque_rew * 0.,
            motor_velocity_rew * 0.02,
            action_smooth_rew * 1.,
            collision_rew * 0.,
            ground_impact_rew * 0.0,
            self_collision_rew * 1,
            motor_error_rew * 0.2,
            pmf_rew * 0.05
        ])
        # rewards[7:] *= self.schedule_ratio
        rewards = np.clip(rewards, a_min=-14, a_max=15) / 100.
        return rewards

    reward_name = [
        'const',
        'r_rate_rew',
        'foot_support',
        'base_height',
        'foot_height',
        'motor_constrain',
        'base_vel',
        'py_rate',
        'foot_slip',
        'foot_vz',
        'motor_torque',
        'motor_velocity',
        'action_smooth',
        'collision_force',
        'ground_impact',
        'self_collision',
        'motor_error',
        'pmf'
    ]

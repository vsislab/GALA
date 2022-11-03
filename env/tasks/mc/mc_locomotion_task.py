# -*- coding: utf-8 -*-
import collections
import enum
from math import sin, cos, pi
import numpy as np
from numpy.linalg import norm

from env.commanders import Commander
from env.tasks.common import register
from env.tasks import BaseMCTask, RecoveryTask
from env.tasks import ForwardLocomotionTask
from env.utils import getMatrixFromEuler, pose3d, PhaseModulator


def _convert_world_to_base_frame(quantity, rpy):
    return np.dot(quantity, getMatrixFromEuler(rpy))


@register
class MCForwardLocomotionTask(BaseMCTask):
    MOTOR_POSITION_HIGH = np.array([1.222, pi, -0.646]).repeat(4)
    MOTOR_POSITION_LOW = np.array([-1.222, -pi, -2.775]).repeat(4)

    config = {'base_frequency': 0.}
    config['action_high'] = np.array([5, 10.6, 10.6, 10.6]).repeat(4)  # incremental:(Vx, Vy, Vz)
    config['action_low'] = np.array([0, -10.6, -10.6, -10.6]).repeat(4)
    config['command_duration_time'] = 5.
    config['forward_velocity_range'] = (0.2, 0.6)
    config['lateral_velocity_range'] = (0.2, 0.6)
    config['body_height_range'] = (0.25, 0.45)
    config['leg_width_ranged'] = (0.03, 0.1)
    config['heading_range'] = (-pi, pi)
    task_adv_coef = [0.6, 0.7]
    task_cls = [RecoveryTask, ForwardLocomotionTask]

    net_id = 0

    def __init__(self, env, **kwargs):
        self.commander = Commander(env,
                                   env.mode,
                                   self.config['command_duration_time'],
                                   self.config['rolling_rate_range'],
                                   self.config['forward_velocity_range'],
                                   self.config['lateral_velocity_range'],
                                   self.config['body_height_range'],
                                   self.config['leg_width_ranged'],
                                   self.config['heading_range'])
        self.terrain_type = env.terrain.param.type.name if env.terrain_randomizer else 'Flat'
        self.pms = [PhaseModulator(time_step=env.control_time_step, f0=self.config['base_frequency']) for _ in range(4)]
        self.motor_position_reference = np.repeat(env.aliengo.STAND_MOTOR_POSITION_REFERENCE, 4)
        self.motor_position_norm = np.repeat([1, 0.8, 1.5], 4)
        self.action_history = collections.deque(maxlen=10)
        self.net_out_history = collections.deque(maxlen=2)
        self.base_flip_history = collections.deque(maxlen=10)
        self.motor_position = None
        self.standing_count = 0
        self.base_flip = 0
        self.ground_impact_force = None
        self.task_id = None
        self.reset_time = 0.
        self.vy_sum = 0

        super(MCForwardLocomotionTask, self).__init__(env)

    def reset(self):
        self.commander.reset()
        self.vy_sum = 0
        pms_phi0 = np.random.uniform(low=-pi, high=pi, size=4)
        for pm, phi0 in zip(self.pms, pms_phi0):
            pm.reset(phi0=phi0)
        self.pm_frequency = np.zeros(4)
        self.pm_phase = np.zeros(4)
        self.current_action = self.env.aliengo.motor_position
        for _ in range(self.action_history.maxlen):
            self.action_history.append(self.current_action)
        for _ in range(self.net_out_history.maxlen):
            self.net_out_history.append(np.zeros(self.action_space.shape))
        self.refresh(reset=True)
        for _ in range(self.base_flip_history.maxlen):
            self.base_flip_history.append(self.base_flip)
        self.standing_count = 0
        if self.debug:
            self._debug_param = {
                'task_id': self.task_id,
                'flip': 0,
                'command': np.zeros(len(self.commander.command)),
                'net_out': np.zeros(self.action_space.shape),
                'PM_phase': np.asarray([pm.phi for pm in self.pms]),
            }

    def refresh(self, reset=False):
        self.task_adv_coef = [1, 1, 1, 1, 1, 1, 1]
        self.commander.refresh()
        if self.base_flip > 0.9 and self.base_height > 0.3:
            self.standing_count += 1
        else:
            self.standing_count = 0
        self.self_contact_state = self.env.get_self_contact_state()
        self.body_contact_state = self.env.get_body_contact_state()
        self.foot_contact_state = self.env.get_foot_contact_state()
        self.foot_scanned_height = self.env.get_scanned_height_around_foot()
        self.foot_height_error = self.env.get_scanned_height_under_foot()
        self.last_ground_impact_force, self.ground_impact_force = self.ground_impact_force, abs(
            self.foot_contact_state['force'])
        self.last_motor_position, self.motor_position = self.motor_position, self.env.aliengo.motor_position
        terrain_height = np.mean([self.env.terrain.get_height(self.env.aliengo.position[:2] + np.array(offset))
                                  for offset in [(0., 0.), (0.1, 0), (0, 0.1), (-0.1, 0), (0, -0.1)]])
        self.base_height = self.env.aliengo.position[2] - terrain_height
        self.base_flip = np.array([0, 0, 1]).dot(
            pose3d.QuaternionRotatePoint(self.env.aliengo.orientation, np.array([0, 0, 1])))
        self.base_flip_history.append(self.base_flip)
        self.task_id = self.net_id
        # command: [yaw_rate, forward velocity, lateral velocity, body height, leg width, selfrighting]
        self.command = [self.commander.yaw_rate,
                        self.commander.forward_velocity,
                        self.commander.lateral_velocity,
                        self.commander.body_height,
                        self.commander.leg_width,
                        self.commander.rolling_rate]
        if reset:
            self.last_ground_impact_force = self.ground_impact_force
            self.last_motor_position = self.motor_position

    def observation(self, obs):
        base_velocity = _convert_world_to_base_frame(obs['velocity'], obs['rpy'])
        base_rpy_rate = _convert_world_to_base_frame(obs['rpy_rate'], obs['rpy'])
        motor_position = obs['motor_position'] / self.motor_position_norm
        self.motor_error = motor_error = obs['motor_position'] - self.current_action
        if self.env.rhy_mask:
            rhythm_mask = 0 if self.task_id == 1 else 1.
        else:
            rhythm_mask = 1.
        if not self.env.training and (norm(self.command) < 1e-1
                                      or ((abs(self.env.aliengo.rpy[0]) > 0.6
                                           or abs(self.env.aliengo.rpy[1]) > 0.6
                                           or (self.command[3] > -0 and self.base_height < 0.25))
                                          and self.commander.rolling_rate == 0)):
            rhythm_mask = 0
            self.command = [0, 0, 0, 0, 0, 0]
        pm_phase = np.concatenate([[sin(pm.phi), cos(pm.phi)] for pm in self.pms])
        foot_contact_mask = self.foot_contact_state['mask'].astype(float)
        state = np.concatenate([
            self.command,
            [self.base_flip_history[4], self.base_flip],
            base_velocity[:2],
            [obs['velocity'][2]],
            getMatrixFromEuler(obs['rpy'])[2],
            base_rpy_rate / 5.,
            motor_position,
            obs['motor_velocity'] / 15.,
            motor_error,
            rhythm_mask * pm_phase,
            rhythm_mask * self.pm_frequency / 4.,
            foot_contact_mask
        ])
        return state

    def action(self, net_out):
        self.net_out_history.append(net_out)
        net_out = self.transform(net_out)
        for i in range(4):
            self.pms[i].compute(net_out[i])
        self.pm_frequency = net_out[:4]
        act = self.current_action + net_out[4:] * self.env.control_time_step  # incremental
        act = np.clip(act, a_min=self.MOTOR_POSITION_LOW, a_max=self.MOTOR_POSITION_HIGH)
        self.current_action = act
        self.action_history.append(act)
        if self.debug:
            self._debug_param.update({
                'task_id': self.task_id,
                'flip': self.base_flip,
                'command': [*self.commander.command * np.array([7, 1, 1, 20, 10, 1])],
                'net_out': net_out,
                'PM_phase': np.asarray([pm.phi for pm in self.pms]),
                'foot_force': self.foot_contact_state['force'].clip(max=200, min=0),
            })
        return act

    def success(self):
        suc = False
        if self.env.push_randomizer:
            self.env.push_randomizer.enable(True)
        return suc

    def terminate(self):
        terminate = self.body_contact_state['num'] > 0 or norm(self.env.aliengo.rpy[0]) > 0.7 or norm(self.env.aliengo.rpy[1]) > 0.7  # \
        suc = self.success()
        done = terminate or suc if self.env.training else False
        info = {
            'success': suc,
            # 'value_mask': not terminate,
            'value_mask': not done,
            'reward_mask': not done,
            'global_mask': not done,
            'task_id': self.task_id if self.env.mc else 0,
        }
        return done, info

    def get_reset_state(self):
        reset_state = {}
        if self.env.mode[0] == 1:
            self.reset_time = 0.  # TODO
            reset_state['reset_time'] = self.reset_time
            reset_state['rpy'] = np.array([0, 0, 0])
            reset_state['rpy_rate'] = np.array([0, 0, 0])
            reset_state['motor_velocity'] = np.array([0, 0, 0]).repeat(4)
            reset_state['reset_motor_position'] = np.repeat([0, 0.8, -1.5], 4)  # stand
            reset_state['position'] = np.array([-5, -8.56, 0.38])
        else:
            self.reset_time = 0.4
            reset_state['reset_time'] = self.reset_time
            reset_state['position'] = np.array([-1., 0., 0.5])
            reset_state['reset_motor_position'] = np.repeat([0, 0.8, -1.5], 4)  # stand
            reset_state['motor_position'] = reset_state['reset_motor_position'] + \
                                            np.random.uniform(low=np.array([-0.1, -0.1, -0.1]).repeat(4),
                                                              high=np.array([0.1, 0.1, 0.1]).repeat(4))
        return reset_state

    def reward(self):
        return self.task_reward_fn[self.task_id](self)

    @property
    def debug_name(self):
        return {
            'task_id': ['task_id'],
            'flip': ['flip'],
            'command': [*self.commander.name],
            'net_out': [f'{l}_{o}' for o in ['f', 'x', 'y', 'z'] for l in self.env.aliengo.leg_names],
            'PM_phase': [str(l) for l in self.env.aliengo.leg_names],
            'foot_force': [str(l) for l in self.env.aliengo.leg_names],
        }


@register
class MCHighSpeedLocomotionTask(MCForwardLocomotionTask):
    pass


@register
class MCLateralLocomotionTask(MCForwardLocomotionTask):
    pass


@register
class MCHeightLocomotionTask(MCForwardLocomotionTask):
    def terminate(self):
        terminate = self.body_contact_state['num'] > 0 or abs(self.env.aliengo.rpy[0]) > 0.7 or abs(
            self.env.aliengo.rpy[1]) > 0.7 or abs(self.env.aliengo.base_velocity[1]) > 0.5  # or \
        suc = self.success()
        done = terminate or suc if self.env.training else False
        info = {
            'success': suc,
            # 'value_mask': not terminate,
            'value_mask': not done,
            'reward_mask': not done,
            'global_mask': not done,
            'task_id': self.task_id if self.env.mc else 0,
        }
        return done, info


@register
class MCWidthLocomotionTask(MCForwardLocomotionTask):
    def terminate(self):
        terminate = self.body_contact_state['num'] > 0 or norm(self.env.aliengo.rpy[0]) > 0.7 or norm(
            self.env.aliengo.rpy[1]) > 0.7 or abs(self.env.aliengo.base_velocity[1]) > 0.5  # or \
        suc = self.success()
        done = terminate or suc if self.env.training else False
        info = {
            'success': suc,
            # 'value_mask': not terminate,
            'value_mask': not done,
            'reward_mask': not done,
            'global_mask': not done,
            'task_id': self.task_id if self.env.mc else 0,
        }
        return done, info


@register
class MCRecoveryTask(MCForwardLocomotionTask):
    def success(self):
        suc = False
        if self.env.push_randomizer:
            self.env.push_randomizer.enable(True)
        return suc

    def terminate(self):
        terminate = self.self_contact_state['num'] > 2 or self.base_height > 1. or norm(self.env.aliengo.base_rpy_rate) > 15 or \
                    abs(self.env.aliengo.rpy[1]) > 0.6 or (self.base_flip_history[4] >= 0.9 and self.base_flip < 0.8)  # or np.any(self.foot_height_error < -0.2)
        suc = self.success()
        done = terminate or suc if self.env.training else False
        info = {
            'success': suc,
            # 'value_mask': not terminate,
            'value_mask': not done,
            'reward_mask': not done,
            'global_mask': not done,
            'task_id': self.task_id if self.env.mc else 0,
        }
        return done, info

    def get_reset_state(self):
        reset_state = {}
        self.reset_time = 2.4
        reset_state['reset_time'] = self.reset_time
        reset_state['reset_motor_position'] = np.repeat([0, 1.4, -2.5], 4)  # fold
        p = np.random.uniform()
        if p < 0.2:
            self.reset_time = 0.
            reset_state['reset_time'] = self.reset_time
            reset_state['position'] = np.array([-1., 0., 0.45])
            reset_state['reset_motor_position'] = np.repeat([0, 0.8, -1.5], 4)  # stand
            reset_state['motor_position'] = reset_state['reset_motor_position'] + \
                                            np.random.uniform(low=np.array([-0.1, -0.1, -0.1]).repeat(4),
                                                              high=np.array([0.1, 0.1, 0.1]).repeat(4))
        else:
            if p <= 0.4:
                self.reset_time = 0.1
                reset_state['position'] = np.array([-1, 0., 0.4])
                reset_state['motor_position'] = reset_state['reset_motor_position'] + \
                                                np.random.uniform(low=np.array([-0.1, -0.1, -0.1]).repeat(4),
                                                                  high=np.array([0.1, 0.1, 0.1]).repeat(4))
            elif p <= 0.7:
                reset_state['position'] = np.array([-1, 0., 0.7])
                reset_state['rpy'] = np.random.uniform(low=[-pi, -0., 0], high=[pi, 0., 0])
                reset_state['motor_position'] = reset_state['reset_motor_position'] + \
                                                np.random.uniform(low=np.array([-0.5, -1, -0.0]).repeat(4),
                                                                  high=np.array([0.5, 1, 1]).repeat(4))
            else:
                reset_state['position'] = np.array([-1, 0., 0.7])
                reset_state['rpy'] = np.random.uniform(low=[-pi, -0., 0], high=[-pi, 0., 0])
                reset_state['motor_position'] = np.random.uniform(low=np.array([-0.6, -0.5 * pi, -2.5]).repeat(4),
                                                                  high=np.array([0.6, 0.5 * pi, -0.4]).repeat(4))
        reset_state['motor_position'] = np.clip(reset_state['motor_position'], a_min=self.MOTOR_POSITION_LOW,
                                                a_max=self.MOTOR_POSITION_HIGH)
        return reset_state


@register
class MCRollingTask(MCForwardLocomotionTask):

    def terminate(self):
        terminate = self.self_contact_state['num'] > 2 or norm(self.env.aliengo.base_rpy_rate[1:]) > 5 or self.base_height > 1  # or \

        suc = self.success()
        done = terminate or suc if self.env.training else False
        info = {
            'success': suc,
            # 'value_mask': not terminate,
            'value_mask': not done,
            'reward_mask': not done,
            'global_mask': not done,
            'task_id': self.task_id if self.env.mc else 0,
        }
        return done, info

    def get_reset_state(self):
        if self.env.terrain_updated:
            self.reset_states.clear()
        reset_state = {}
        self.reset_time = 0.15
        reset_state['reset_time'] = self.reset_time
        reset_state['reset_motor_position'] = np.repeat([0, 1.4, -2.5], 4)  # fold
        p = np.random.uniform()
        if p <= 0.5:
            reset_state['reset_time'] = self.reset_time
            reset_state['position'] = np.array([-1., 0., 0.6])
            reset_state['motor_position'] = np.repeat([0, 0.8, -1.5], 4) + \
                                            np.random.uniform(low=np.array([-0.3, -0.3, -0.3]).repeat(4),
                                                              high=np.array([0.3, 0.3, 0.3]).repeat(4))
        else:
            reset_state['position'] = np.array([-1, 0., 0.7])
            reset_state['rpy'] = np.random.uniform(low=[-pi, -0., 0], high=[pi, 0., 0])
            reset_state['motor_position'] = reset_state['reset_motor_position'] + \
                                            np.random.uniform(low=np.array([-0.5, -1, -0]).repeat(4),
                                                              high=np.array([0.5, 1, 1]).repeat(4))
        reset_state['motor_position'] = np.clip(reset_state['motor_position'], a_min=self.MOTOR_POSITION_LOW,
                                                a_max=self.MOTOR_POSITION_HIGH)
        return reset_state


@register
class MCMixTask(MCForwardLocomotionTask):
    pass

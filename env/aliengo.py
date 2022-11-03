# -*- coding: utf-8 -*-
from math import sqrt, pi, cos
import numpy as np
import collections

from .motor import Motor
from .utils import convert_world_to_base_frame, _get_right_size_value

LEG_NUM = FOOT_NUM = 4
MOTOR_NUM = 12
LEG_LENGTH = np.array([0.083, 0.25, 0.25])
MOTOR_FORCE_LIMIT = np.array([44, 44, 55]).repeat(4)
MOTOR_VELOCITY_LIMIT = np.array([20, 20, 16]).repeat(4)
HALF_BODY_LENGTH = 0.2399  # m
BODY_LENGTH = 0.2399 * 2  # m

LEG_NAMES = ['RF', 'LF', 'RR', 'LR']  # match with the order of the real aliengo.
PAYLOAD_NAME = 'payload_fixed'
TRUNK_NAMES = ['trunk_fixed']
# TRUNK_NAMES = ['trunk_motor_fixed', 'trunk_power_fixed', 'trunk_board_fixed']
HIP_NAMES = [f'{leg}_hip_joint' for leg in LEG_NAMES]
SHOULDER_NAMES = [f'{leg}_thigh_joint' for leg in LEG_NAMES]
# THIGH_NAMES = [f'{leg}_thigh_fixed' for leg in LEG_NAMES]
CALF_NAMES = [f'{leg}_calf_joint' for leg in LEG_NAMES]
FOOT_NAMES = [f'{leg}_foot_fixed' for leg in LEG_NAMES]
MOTOR_NAMES = np.concatenate([HIP_NAMES, SHOULDER_NAMES, CALF_NAMES])
OBSERVATION_NAME = ('position', 'velocity', 'rpy', 'rpy_rate', 'motor_position', 'motor_velocity', 'motor_torque')
OBSERVATION_NOISE = {name: 0. for name in OBSERVATION_NAME}


def _parse_observation_noise_dict(noise_dict):
    noise = np.zeros(48, dtype=float)
    noise[:3] = noise_dict['position']
    noise[3:6] = noise_dict['velocity']
    noise[6:9] = noise_dict['rpy']
    noise[9:12] = noise_dict['rpy_rate']
    noise[12:24] = noise_dict['motor_position']
    noise[24:36] = noise_dict['motor_velocity']
    noise[36:48] = noise_dict['motor_torque']
    return noise


class Aliengo:
    DEFAULT_INIT_POSITION = np.array([0., 0., 0.4])
    DEFAULT_INIT_RACK_POSITION = np.array([0., 0., 1.0])
    DEFAULT_INIT_VELOCITY = np.array([0., 0., 0.])
    DEFAULT_INIT_RPY = np.array([0., 0., 0.])
    DEFAULT_INIT_RPY_RATE = np.array([0., 0., 0.])
    STAND_MOTOR_POSITION_REFERENCE = np.array([0., 0.8, -1.5])
    FOLD_MOTOR_POSITION_REFERENCE = np.array([0, 1.4, -2.5])
    DEFAULT_INIT_MOTOR_POSITION = STAND_MOTOR_POSITION_REFERENCE.repeat(LEG_NUM)
    DEFAULT_INIT_MOTOR_VELOCITY = np.zeros(MOTOR_NUM, dtype=np.float32)

    FOOT_POSITION_REFERENCE = np.asarray([LEG_LENGTH[0], (LEG_LENGTH[2] - LEG_LENGTH[1]) / sqrt(2), -sqrt(
        LEG_LENGTH[1] ** 2 + LEG_LENGTH[2] ** 2 - 2 * LEG_LENGTH[1] * LEG_LENGTH[2] * cos(pi - abs(STAND_MOTOR_POSITION_REFERENCE[2])))])

    def __init__(self,
                 urdf_file,
                 client,
                 time_step=0.001,
                 action_repeat=1,
                 observation_noise=None,
                 remove_default_link_damping=False,
                 self_collision_enabled=False,
                 kp=1.,
                 kd=0.,
                 motor_damping=0.,
                 motor_friction=0.,
                 foot_friction=0.5,
                 foot_restitution=0.01,
                 latency=0.0,
                 on_rack=False):
        self.client = client
        self._urdf_file = urdf_file
        self._time_step = time_step
        self._action_repeat = action_repeat
        default_observation_noise = OBSERVATION_NOISE.copy()
        if observation_noise is not None:
            assert isinstance(observation_noise, dict), observation_noise
            for name in observation_noise:
                assert name in default_observation_noise
                default_observation_noise[name] = abs(observation_noise[name])
        self.default_observation_noise = default_observation_noise.copy()
        self.observation_noise = default_observation_noise.copy()
        self._remove_default_link_damping = remove_default_link_damping
        self._self_collision_enabled = self_collision_enabled
        self._kp = _get_right_size_value(kp, self.motor_num)
        self._kd = _get_right_size_value(kd, self.motor_num)
        self.motor = Motor(self._kp, self._kd, strength_ratio=1., damping=motor_damping, dry_friction=motor_friction)
        self._foot_friction = foot_friction
        self._foot_restitution = foot_restitution
        self._latency = latency
        self._on_rack = on_rack
        self.observation_history = collections.deque(maxlen=int(0.1 / time_step))
        self.action_history = collections.deque(maxlen=int(0.1 / time_step))
        self._init_position = None
        self._init_orientation = None
        self._urdf_id = None
        self._param = {'mass': None,
                       'inertia': None,
                       'payload': None,
                       'latency': latency,
                       'motor strength': self.motor.strength_ratio,
                       'motor friction': self.motor.dry_friction,
                       'motor damping': self.motor.damping,
                       }
        # self.action_filter = ActionFilterButter(sampling_rate=1 / (self._time_step * self._action_repeat),
        #                                         num_joints=self.motor_num) if action_filter_enabled else None
        self._reload_urdf = True
        self.reset()

    def reset(self,
              position=None,
              velocity=None,
              rpy=None,
              rpy_rate=None,
              motor_position=None,
              motor_velocity=None,
              reset_motor_position=None,
              reset_time=0.):
        if position is None: position = self.default_init_position
        if velocity is None: velocity = self.DEFAULT_INIT_VELOCITY
        if rpy is None: rpy = self.DEFAULT_INIT_RPY
        if rpy_rate is None: rpy_rate = self.DEFAULT_INIT_RPY_RATE
        if motor_position is None: motor_position = self.DEFAULT_INIT_MOTOR_POSITION
        if motor_velocity is None: motor_velocity = self.DEFAULT_INIT_MOTOR_VELOCITY
        self._init_position = position
        self._init_orientation = init_orientation = self.client.getQuaternionFromEuler(rpy)
        self._init_motor_position = motor_position
        if self._reload_urdf:
            self._reload_urdf = False
            self._urdf_id = self.client.loadURDF(
                self._urdf_file, position, init_orientation,
                useFixedBase=self._on_rack,
                flags=(self.client.URDF_USE_SELF_COLLISION if self._self_collision_enabled else 0)
                # | self.client.URDF_MERGE_FIXED_LINKS
                # |self.client.URDF_USE_INERTIA_FROM_FILE
            )
            self._record_info_from_urdf()
            # self.set_foot_spinning_friction(0.0065) # TODO
            self.set_foot_lateral_friction(self._foot_friction)
            self.set_foot_restitution(self._foot_restitution)
            self.set_motor_velocity_limit(MOTOR_VELOCITY_LIMIT)
            self.disable_default_motor()
            if self._remove_default_link_damping:
                self.remove_default_link_damping()

        self.client.resetBasePositionAndOrientation(self._urdf_id, position, init_orientation)
        self.client.resetBaseVelocity(self._urdf_id, velocity, rpy_rate)
        if reset_time <= 0.:
            reset_motor_position = motor_position
        elif reset_motor_position is None:
            reset_motor_position = self.DEFAULT_INIT_MOTOR_POSITION
        for motor_id, pos, vel in zip(self._motor_id_list, reset_motor_position, motor_velocity):
            self.client.resetJointState(self._urdf_id, motor_id, targetValue=pos, targetVelocity=vel)
        self.client.stepSimulation()
        if self.self_collision: return

        self.settle_down_for_reset(reset_time, reset_motor_position, motor_position)
        self.refresh()
        self.motor_torque = np.zeros(self.motor_num, dtype=float)
        self.observation_history.clear()
        self.action_history.clear()
        return self.receive_observation()
        # if self.action_filter:
        #     self.action_filter.reset()
        #     self.action_filter.init_history(init_motor_position)

    def settle_down_for_reset(self, reset_time, init_motor_position, target_motor_position):
        # Perform reset motion within reset_time
        if reset_time <= 1e-2: return
        iteration = max(int(reset_time / self._time_step), 1)
        delta_motor_position = (target_motor_position - init_motor_position) / iteration
        self.client.setPhysicsEngineParameter(enableConeFriction=0, numSolverIterations=100, numSubSteps=2)
        for i in range(iteration):
            self.apply_action(init_motor_position + i * delta_motor_position)
            self.client.stepSimulation()
            # [yaw, pitch, dist] = self.client.getDebugVisualizerCamera()[8:11]
            # self.client.resetDebugVisualizerCamera(dist, yaw, pitch, self.position)
            # self.client.configureDebugVisualizer(self.client.COV_ENABLE_SINGLE_STEP_RENDERING, 1)  # smooth simulation rendering

    def _record_info_from_urdf(self):
        joint_dict = {}
        for i in range(self.client.getNumJoints(self._urdf_id)):
            joint_info = self.client.getJointInfo(self._urdf_id, i)
            joint_dict[joint_info[1].decode('UTF-8')] = joint_info[0]
        self._motor_id_list = [joint_dict[name] for name in MOTOR_NAMES]

        self._trunk_id_list = [joint_dict[name] for name in TRUNK_NAMES]
        self._hip_id_list = [joint_dict[name] for name in HIP_NAMES]
        self._shoulder_id_list = [joint_dict[name] for name in SHOULDER_NAMES]
        # self._thigh_id_list = [joint_dict[name] for name in THIGH_NAMES]
        self._calf_id_list = [joint_dict[name] for name in CALF_NAMES]
        self._foot_id_list = [joint_dict[name] for name in FOOT_NAMES]
        self._link_id_list = [self._trunk_id_list, self._hip_id_list, self._shoulder_id_list,
                              self._calf_id_list, self._foot_id_list]
        # self._link_id_list = [self._trunk_id_list, self._hip_id_list, self._shoulder_id_list,
        #                       self._thigh_id_list, self._calf_id_list, self._foot_id_list]
        self._link_mass = [np.array([self.client.getDynamicsInfo(self._urdf_id, i)[0] for i in id_list])
                           for id_list in self._link_id_list]
        self._link_inertia = [np.array([self.client.getDynamicsInfo(self._urdf_id, i)[2] for i in id_list])
                              for id_list in self._link_id_list]
        self._param['mass'] = np.concatenate(self._link_mass, axis=0)
        self._param['inertia'] = np.concatenate(self._link_inertia, axis=0)
        # self._payload_id = joint_dict[PAYLOAD_NAME]
        # self._param['payload'] = self.client.getDynamicsInfo(self._urdf_id, self._payload_id)[0]
        # self._param['com'] = np.array(self.client.getLinkState(self._urdf_id, self._trunk_id_list[0])[2])

    def refresh(self):
        position, orientation = self.client.getBasePositionAndOrientation(self._urdf_id)
        velocity, rpy_rate = self.client.getBaseVelocity(self._urdf_id)
        self.position = np.asarray(position)
        self.velocity = np.asarray(velocity)
        self.orientation = np.asarray(orientation)
        self.rpy_rate = np.asarray(rpy_rate)
        self.rpy = np.asarray(self.client.getEulerFromQuaternion(orientation))
        self.rotation_matrix = np.reshape(self.client.getMatrixFromQuaternion(orientation), (3, 3), order='C')
        self.base_velocity = np.dot(self.velocity, self.rotation_matrix)
        self.base_rpy_rate = np.dot(self.rpy_rate, self.rotation_matrix)

        motor_state = self.client.getJointStates(self._urdf_id, self._motor_id_list)
        self.motor_position = np.asarray([x[0] for x in motor_state])
        self.motor_velocity = np.asarray([x[1] for x in motor_state])

        foot_state = self.client.getLinkStates(self._urdf_id, self._foot_id_list, computeForwardKinematics=True, computeLinkVelocity=True)
        self.foot_position = np.asarray([x[0] for x in foot_state])
        self.foot_velocity = np.asarray([x[6] for x in foot_state])

        self.base_foot_position = np.dot(self.foot_position, self.rotation_matrix)

    def receive_observation(self):
        self.refresh()
        obs = np.concatenate([self.position, self.velocity, self.orientation, self.rpy_rate, self.motor_position, self.motor_velocity, self.motor_torque])
        self.observation_history.appendleft(obs)
        value_history = self.observation_history
        time_interval = self._time_step * self._action_repeat
        latency = self._latency + 0.01
        n_steps_ago = int(latency / time_interval)
        if n_steps_ago + 1 >= len(value_history):
            delayed_obs = value_history[-1]
        else:
            remaining_latency = latency - n_steps_ago * time_interval
            blend_alpha = remaining_latency / time_interval
            delayed_obs = ((1.0 - blend_alpha) * np.asarray(value_history[n_steps_ago]) +
                           blend_alpha * np.asarray(value_history[n_steps_ago + 1]))
        delayed_obs = np.concatenate([delayed_obs[:6],
                                      self.client.getEulerFromQuaternion(delayed_obs[6:10]),
                                      delayed_obs[10:]])
        return {
            'position': delayed_obs[0:3] + np.random.uniform(-self.observation_noise['position'], self.observation_noise['position'], 3),
            'velocity': delayed_obs[3:6] + np.random.uniform(-self.observation_noise['velocity'], self.observation_noise['velocity'], 3),
            'rpy': delayed_obs[6:9] + np.random.uniform(-self.observation_noise['rpy'], self.observation_noise['rpy'], 3),
            'rpy_rate': delayed_obs[9:12] + np.random.uniform(-self.observation_noise['rpy_rate'], self.observation_noise['rpy_rate'], 3),
            'motor_position': delayed_obs[12:24] + np.random.uniform(-self.observation_noise['motor_position'], self.observation_noise['motor_position'], 12),
            'motor_velocity': delayed_obs[24:36] + np.random.uniform(-self.observation_noise['motor_velocity'], self.observation_noise['motor_velocity'], 12),
            'motor_torque': delayed_obs[36:48] + np.random.uniform(-self.observation_noise['motor_torque'], self.observation_noise['motor_torque'], 12)
        }

    def receive_action(self, action):
        self.action_history.appendleft(action)
        time_interval = self._time_step * self._action_repeat
        n_steps_ago = int(self._latency / time_interval) + 1
        n_steps_ago = min(n_steps_ago, len(self.action_history) - 1)
        return self.action_history[n_steps_ago]

    def apply_action(self, action):
        motor_state = self.client.getJointStates(self._urdf_id, self._motor_id_list)
        motor_position = np.asarray([x[0] for x in motor_state])
        motor_velocity = np.asarray([x[1] for x in motor_state])
        motor_position = motor_position + np.random.uniform(-self.observation_noise['motor_position'], self.observation_noise['motor_position'], 12)
        motor_velocity = motor_velocity + np.random.uniform(-self.observation_noise['motor_velocity'], self.observation_noise['motor_velocity'], 12)
        self.motor_torque = np.clip(self.motor.convert_to_torque(action, motor_position, motor_velocity), -MOTOR_FORCE_LIMIT, MOTOR_FORCE_LIMIT)
        self.client.setJointMotorControlArray(bodyIndex=self._urdf_id,
                                              jointIndices=self._motor_id_list,
                                              controlMode=self.client.TORQUE_CONTROL,
                                              forces=self.motor_torque)

    def terminate(self):
        pass

    def is_fallen(self, threshold=0.5):
        return self.rotation_matrix[-1, -1] < threshold

    def draw_foot_trajectory(self, init=False):
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]
        if not init:
            for i in range(self.leg_num):
                self.client.addUserDebugLine(self._prev_foot_position[i], self.foot_position[i], colors[i], lifeTime=1.)
        self._prev_foot_position = self.foot_position

    def get_link_position(self, link_ids):
        return np.asarray(
            [link_info[0] for link_info in
             self.client.getLinkStates(self._urdf_id, link_ids, computeForwardKinematics=True)])

    def get_foot_position_on_base_frame(self):
        pos = self.foot_position - self.get_link_position(self._hip_id_list)
        return convert_world_to_base_frame(pos, self.rpy)

    def remove_default_link_damping(self):
        # remove damp and refer to Page 85 of "pybullet quick start"
        for l in self.link_id:
            self.client.changeDynamics(self._urdf_id, l, linearDamping=0, angularDamping=0)

    def disable_default_motor(self):
        # disabled default motors in pybullet by setting friction to 0
        for motor_id in self._motor_id_list:
            self.client.setJointMotorControl2(
                bodyIndex=self._urdf_id,
                jointIndex=motor_id,
                controlMode=self.client.VELOCITY_CONTROL,
                targetVelocity=0,
                force=0.)

    def set_motor_velocity_limit(self, max_velocity):
        for motor_id, v in zip(self._motor_id_list, max_velocity):
            self.client.changeDynamics(self._urdf_id, motor_id, maxJointVelocity=v)

    def set_foot_lateral_friction(self, friction):
        for i in self._foot_id_list:
            self.client.changeDynamics(self._urdf_id, i, lateralFriction=friction)

    def set_foot_spinning_friction(self, friction):
        for i in self._foot_id_list:
            self.client.changeDynamics(self._urdf_id, i, spinningFriction=friction)

    def get_foot_lateral_friction(self):
        return np.array([self.client.getDynamicsInfo(self._urdf_id, i)[1] for i in self._foot_id_list])

    def get_foot_spinning_friction(self):
        return np.array([self.client.getDynamicsInfo(self._urdf_id, i)[7] for i in self._foot_id_list])

    def set_foot_restitution(self, restitution):
        for i in self._foot_id_list:
            self.client.changeDynamics(self._urdf_id, i, restitution=restitution)

    def get_foot_restitution(self):
        return np.array([self.client.getDynamicsInfo(self._urdf_id, i)[5] for i in self._foot_id_list])

    def get_mass_from_urdf(self):
        return self._link_mass

    def set_mass(self, mass):
        for l, v in zip(self.link_id, mass):
            self.client.changeDynamics(self._urdf_id, l, mass=v)

    def get_inertia_from_urdf(self):
        return self._link_inertia

    def set_inertia(self, inertia):
        for l, v in zip(self.link_id, inertia):
            self.client.changeDynamics(self._urdf_id, l, localInertiaDiagonal=v)

    def set_payload(self, mass):
        self.client.changeDynamics(self._urdf_id, self._payload_id, mass=mass)

    def set_latency(self, latency):
        self._latency = latency

    def set_motor_strength_ratio(self, ratio):
        self.motor.strength_ratio = ratio

    def set_motor_damping(self, damping):
        self.motor.damping = damping

    def set_motor_friction(self, friction):
        self.motor.dry_friction = friction

    @property
    def self_collision(self):
        return len(self.client.getContactPoints(bodyA=self._urdf_id, bodyB=self._urdf_id)) > 0

    @property
    def physics_param(self):
        return self._param

    @property
    def default_init_position(self):
        return np.copy(self.DEFAULT_INIT_POSITION if not self._on_rack else self.DEFAULT_INIT_RACK_POSITION)

    @property
    def time_step(self):
        return self._time_step

    @property
    def motor_num(self):
        return MOTOR_NUM

    @property
    def leg_num(self):
        return LEG_NUM

    @property
    def foot_num(self):
        return FOOT_NUM

    @property
    def leg_length(self):
        return LEG_LENGTH

    @property
    def observation_name(self):
        return OBSERVATION_NAME

    @property
    def leg_names(self):
        return LEG_NAMES

    @property
    def motor_names(self):
        return MOTOR_NAMES

    @property
    def init_position(self):
        return self._init_position

    @property
    def init_orientation(self):
        return self._init_orientation

    @property
    def init_motor_position(self):
        return self._init_motor_position

    @property
    def urdf_file(self):
        return self._urdf_file

    @urdf_file.setter
    def urdf_file(self, f: str):
        self._urdf_file = f
        self._reload_urdf = True
        self.client.removeBody(self._urdf_id)

    @property
    def urdf_id(self):
        return self._urdf_id

    @property
    def link_id_map(self):
        return {
            'trunk': self._trunk_id_list,
            'hip': self._hip_id_list,
            'shoulder': self._shoulder_id_list,
            # 'thigh': self._thigh_id_list,
            'calf': self._calf_id_list,
            'foot': self._foot_id_list
        }

    @property
    def link_id(self):
        return np.concatenate(self._link_id_list)

    @property
    def trunk_id(self):
        return np.asarray(self._trunk_id_list)

    @property
    def hip_id(self):
        return np.asarray(self._hip_id_list)

    @property
    def shoulder_id(self):
        return np.asarray(self._shoulder_id_list)

    # @property
    # def thigh_id(self):
    #     return np.asarray(self._thigh_id_list)

    @property
    def calf_id(self):
        return np.asarray(self._calf_id_list)

    @property
    def foot_id(self):
        return np.asarray(self._foot_id_list)

    @property
    def state_dict(self):
        return {
            'position': self.position,
            'velocity': self.velocity,
            'rpy': self.rpy,
            'rpy_rate': self.rpy_rate,
            'motor_position': self.motor_position,
            'motor_velocity': self.motor_velocity,
        }

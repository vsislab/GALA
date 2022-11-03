# -*- coding: utf-8 -*-
"""Note:
        This environment is just a benchmark, and it can be trained only when combined with the wrapper function.
        You can use it to test the traditional algorithms.
"""
import collections
import gym
from gym import spaces
from gym.utils import seeding
from math import inf, pi, sin, cos
import numpy as np
from os.path import join, dirname
import pybullet
import pybullet_data
import pybullet_utils.bullet_client
import time
from typing import Union

from .aliengo import Aliengo
from .randomizers import ParamRandomizerFromConfig, PushRandomizer, UrdfRandomizer
from .randomizers import TerrainRandomizer, TerrainInstance
from .randomizers import BoxRandomizer

# CAM_DIST = 6.3
# CAM_YAW = -0
# CAM_PITCH = -45
CAM_DIST = 1.5
CAM_YAW = 90
CAM_PITCH = -20
bodyUniqueIdB, linkIndexA, contactNormalOnB, normalForce = 2, 3, 7, 9


class AliengoGymEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 100}

    def __init__(self,
                 seed=None,
                 self_collision_enabled=False,
                 remove_default_link_damping=False,
                 kp=1.0,
                 kd=0.02,
                 motor_damping=0.,
                 motor_friction=0.,
                 foot_friction=1.,
                 foot_restitution=0.1,
                 terrain_friction=1.,
                 terrain_restitution=0.1,
                 observation_noise=None,
                 latency=0.0,
                 time_step=1e-3,
                 action_repeat=1,
                 max_time=float('inf'),
                 on_rack=False,
                 render=False,
                 training=False,
                 mc=True,
                 rhy_mask=True,
                 mode=2,
                 video_path=None,
                 timing_path=None,
                 draw_foot_traj=False,
                 terrain_randomizer: TerrainRandomizer = None,
                 urdf_randomizer: UrdfRandomizer = None,
                 param_randomizer: ParamRandomizerFromConfig = None,
                 push_randomizer: PushRandomizer = None):
        self._time_step = time_step
        self._action_repeat = action_repeat
        self._control_time_step = time_step * action_repeat
        self._time_duration = time_step * action_repeat
        self._self_collision_enabled = self_collision_enabled
        self._remove_default_link_damping = remove_default_link_damping
        self._kp = kp
        self._kd = kd
        self._motor_damping = motor_damping
        self._motor_friction = motor_friction
        self._foot_friction = foot_friction
        self._foot_restitution = foot_restitution
        self._terrain_friction = terrain_friction
        self._terrain_restitution = terrain_restitution
        self._observation_noise = observation_noise
        self._latency = latency
        self.max_time = max_time
        self.training = training
        self.mc = mc
        self.rhy_mask = rhy_mask,
        self.mode = mode,
        self._is_render = render
        self._on_rack = on_rack
        self._video_path = video_path
        self._timing_path = timing_path
        self._is_draw_foot_traj = render and draw_foot_traj
        self.terrain_randomizer = terrain_randomizer
        self.urdf_randomizer = urdf_randomizer
        self.param_randomizer = param_randomizer
        self.push_randomizer = push_randomizer
        self._cam_dist = CAM_DIST
        self._cam_yaw = CAM_YAW
        self._cam_pitch = CAM_PITCH
        self._high_performance_mode = None
        self._last_frame_time = None
        self.aliengo = None
        self.terrain = None
        self.viewer = None
        self._history_size = int(2 / (action_repeat * time_step))
        self._reset_count = 0
        self._counter = 0
        self.objects = {}
        if self._is_render:
            self.client = pybullet_utils.bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self.client = pybullet_utils.bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
        self.client.setAdditionalSearchPath(join(dirname(__file__), 'assets/aliengo/urdf'))
        self.seed(seed)
        self.reset(hard_reset=True)
        # following need to be after reset()
        # self._build_observation_space()
        # self._build_action_space()

    def reset(self, hard_reset=False, **kwargs):
        self._hard_reset = hard_reset
        if self._is_render:
            self.client.configureDebugVisualizer(self.client.COV_ENABLE_RENDERING, 0)
        # ***************** START: hard reset ************************
        if hard_reset or self.terrain_updated:
            self.client.resetSimulation()
            # self.setPhysicsEngineParameter()
            self.client.setTimeStep(self._time_step)
            self.client.setGravity(0, 0, -9.8)
            self.client.configureDebugVisualizer(self.client.COV_ENABLE_GUI, 0)
            # self.client.configureDebugVisualizer(self.client.COV_ENABLE_DEPTH_BUFFER_PREVIEW,0)
            # self.client.configureDebugVisualizer(self.client.COV_ENABLE_RGB_BUFFER_PREVIEW,0)
            # self.client.configureDebugVisualizer(self.client.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,0)
            self.aliengo = Aliengo(urdf_file="aliengo.urdf",
                                   client=self.client,
                                   time_step=self._time_step,
                                   action_repeat=self._action_repeat,
                                   remove_default_link_damping=self._remove_default_link_damping,
                                   self_collision_enabled=self._self_collision_enabled,
                                   kp=self._kp,
                                   kd=self._kd,
                                   motor_damping=self._motor_damping,
                                   motor_friction=self._motor_friction,
                                   foot_friction=self._foot_friction,
                                   foot_restitution=self._foot_restitution,
                                   observation_noise=self._observation_noise,
                                   latency=self._latency,
                                   on_rack=self._on_rack)
            if self.terrain_randomizer:
                self.terrain_randomizer.randomize_env(self)
                self.terrain = self.terrain_randomizer.terrain
            else:
                terrain_id = self.client.loadURDF("%s/plane.urdf" % pybullet_data.getDataPath())
                self.terrain = TerrainInstance(self.client, terrain_id, position=(0., 0., 0.))
            self.terrain.friction = self._terrain_friction
            self.terrain.restitution = self._terrain_restitution
            if self.param_randomizer:
                self.param_randomizer.init_param({**self.aliengo.physics_param,
                                                  'control time step': self._control_time_step,
                                                  'terrain friction': self.terrain.friction,
                                                  'terrain restitution': self.terrain.restitution})
            if self._is_render and self._video_path is not None:
                self._video_id = self.client.startStateLogging(self.client.STATE_LOGGING_VIDEO_MP4, self._video_path)
            if self._timing_path is not None:
                self._timing_id = self.client.startStateLogging(self.client.STATE_LOGGING_PROFILE_TIMINGS, self._timing_path)
        # ***************** END: hard reset ************************
        if self.urdf_randomizer:
            self.urdf_randomizer.randomize_env(self)
        if self.terrain_randomizer:
            self.terrain_randomizer.randomize_env(self)
            if 'position' not in kwargs:
                kwargs['position'] = self.aliengo.default_init_position
            kwargs['position'] += self.terrain.place_position
        if self.param_randomizer:
            self.param_randomizer.randomize_env(self)
        if self.push_randomizer:
            self.push_randomizer.randomize_env(self)
        for name, value in self.objects.items():
            self.client.resetBasePositionAndOrientation(value['id'], value['position'], value['orientation'])
            self.client.resetBaseVelocity(value['id'], value['velocity'], value['rpy_rate'])
        obs = self.aliengo.reset(**kwargs)
        # self.client.stepSimulation()
        if self._is_render:
            self.client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0, 0, 0])
            self.client.configureDebugVisualizer(self.client.COV_ENABLE_RENDERING, 1)
            if self._is_draw_foot_traj:
                self.aliengo.draw_foot_trajectory(init=True)
        self._last_frame_time = time.time()
        self._reset_count += 1
        self._counter = 0
        return obs

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # assert self.action_space.contains(action), action
        if self._is_render:
            # Sleep, otherwise the computation takes less time than real time,
            # which will make the visualization like a fast-forward video.
            time_spent = time.time() - self._last_frame_time
            time_to_sleep = self.control_time_step - time_spent
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
            self._last_frame_time = time.time()
            # Keep the previous orientation of the camera set by the user.
            [yaw, pitch, dist] = self.client.getDebugVisualizerCamera()[8:11]
            self.client.resetDebugVisualizerCamera(dist, yaw, pitch, self.aliengo.position)  # todo: followed camera
            # self.client.resetDebugVisualizerCamera(dist, -45 + abs(self.aliengo.rpy[2]) * 57.3, pitch, self.aliengo.position)  # todo: followed camera
            # self.client.resetDebugVisualizerCamera(dist, yaw, pitch, [0.3, 7., 0])  # todo: fixed camera
            self.client.configureDebugVisualizer(self.client.COV_ENABLE_SINGLE_STEP_RENDERING, 1)  # smooth simulation rendering

        self.setPhysicsEngineParameter()
        act = self.aliengo.receive_action(action)
        for _ in range(self._action_repeat):

            self.aliengo.apply_action(act)
            if self.push_randomizer:
                self.push_randomizer.randomize_step(self)
            self.client.stepSimulation()

        obs = self.aliengo.receive_observation()
        if self._is_draw_foot_traj:
            self.aliengo.draw_foot_trajectory()
        self._counter += 1
        time_limit = self.time >= self.max_time
        space_limit = not self.terrain.in_terrain(self.aliengo.position)
        info = {'time_limit': time_limit, 'space_limit': space_limit}
        return obs, 0., time_limit or space_limit, info

    def fixed_render(self, mode="rgb_array"):
        RENDER_HEIGHT = 2000
        RENDER_WIDTH = 4000
        if mode != "rgb_array":
            return np.array([])
        view_matrix = self.client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.3, 7.5, 0],  # self.aliengo.position, todo
            distance=self._cam_dist + 1.3,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = self.client.computeProjectionMatrixFOV(
            fov=60, aspect=float(RENDER_WIDTH) / RENDER_HEIGHT, nearVal=0.1, farVal=100.0)
        (_, _, px, _, _) = self.client.getCameraImage(
            width=RENDER_WIDTH,
            height=RENDER_HEIGHT,
            renderer=self.client.ER_BULLET_HARDWARE_OPENGL,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix)
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def render(self, mode="fixed", dyaw=0):
        if mode == "fixed":
            RENDER_HEIGHT = 2000
            RENDER_WIDTH = 4000
            camera_position = [0.3, 7.5, 0]
            dist = 6.3
            pitch = -45
            yaw = 0
        else:
            RENDER_HEIGHT = 760
            RENDER_WIDTH = 760
            camera_position = self.aliengo.position
            dist = 1.3
            pitch = -20
            yaw = dyaw + abs(self.aliengo.rpy[2])

        view_matrix = self.client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=camera_position,  # todo
            distance=dist,
            yaw=yaw,
            pitch=pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = self.client.computeProjectionMatrixFOV(
            fov=60, aspect=float(RENDER_WIDTH) / RENDER_HEIGHT, nearVal=0.1, farVal=100.0)
        (_, _, px, _, _) = self.client.getCameraImage(
            width=RENDER_WIDTH,
            height=RENDER_HEIGHT,
            renderer=self.client.ER_BULLET_HARDWARE_OPENGL,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix)
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def close(self):
        if self._is_render and self._video_path is not None:
            self.client.stopStateLogging(self._video_id)
        if self._timing_path is not None:
            self.client.stopStateLogging(self._timing_id)
        self.client.disconnect()

    def add_object(self,
                   fn: collections.Callable,
                   name: str = None,
                   position: tuple = (0, 0, 0),
                   orientation: tuple = (0, 0, 0, 1),
                   velocity: tuple = (0, 0, 0),
                   rpy_rate: tuple = (0, 0, 0)):
        id = fn(self.client)
        name = str(id) if name is None else str(name)
        self.objects[name] = {
            'id': id,
            'position': position,
            'orientation': orientation,
            'velocity': velocity,
            'rpy_rate': rpy_rate
        }

    def remove_object(self, name: str):
        id = self.objects[name]
        self.client.removeBody(id)
        del self.objects[id]

    def _build_observation_space(self):
        motor_num = self.aliengo.motor_num
        upper_bound = np.zeros(3 + 3 + 3 + 3 + motor_num + motor_num + motor_num, dtype=np.float32)
        upper_bound[0:3] = inf  # position
        upper_bound[3:6] = inf  # velocity
        upper_bound[6:9] = pi  # rpy
        upper_bound[9:12] = inf  # rpy rate
        upper_bound[12:12 + motor_num] = pi  # joint angle
        upper_bound[12 + motor_num:12 + 2 * motor_num] = inf  # joint velocity
        upper_bound[12 + 2 * motor_num:12 + 3 * motor_num] = inf  # joint torque
        self.observation_space = spaces.Box(-upper_bound, upper_bound)

    def _build_action_space(self):
        leg_num = self.aliengo.leg_num
        lower_bound = np.array([-pi, -pi, -pi], dtype=np.float32).repeat(leg_num)
        upper_bound = np.array([pi, pi, pi], dtype=np.float32).repeat(leg_num)
        self.action_space = spaces.Box(lower_bound, upper_bound)

    def get_self_contact_points(self):
        urdf_id = self.aliengo.urdf_id
        return self.client.getContactPoints(bodyA=urdf_id, bodyB=urdf_id)

    def get_self_contact_state(self):
        contact_ids, contact_forces = [], []
        for p in self.get_self_contact_points():
            if p[normalForce] > 0.:
                contact_ids.append(p[linkIndexA])
                contact_forces.append(p[normalForce])
        return {'id': contact_ids,
                'num': len(contact_ids),
                'force': contact_forces}

    def get_contact_points(self):
        urdf_id = self.aliengo.urdf_id
        return tuple(c for c in self.client.getContactPoints(bodyA=urdf_id) if c[bodyUniqueIdB] != urdf_id)

    def get_body_contact_state(self, exclude_body_names: list = None):
        if exclude_body_names is None:
            exclude_body_names = []
        exclude_body_names.append('foot')
        exclude_body_contact_ids = np.concatenate([self.aliengo.link_id_map[name] for name in set(exclude_body_names)])
        contact_ids, contact_forces = [], []
        for p in self.get_contact_points():
            if p[linkIndexA] in exclude_body_contact_ids:
                continue
            if p[normalForce] > 0.:
                contact_ids.append(p[linkIndexA])
                contact_forces.append(p[normalForce])
        return {
            'id': contact_ids,
            'num': len(contact_ids),
            'force': contact_forces
        }

    def get_foot_contact_state(self):
        include_id_list = self.aliengo._foot_id_list
        contact_points = [[] for _ in range(len(include_id_list))]
        for point in self.get_contact_points():
            ind = point[linkIndexA]
            if ind in include_id_list:
                contact_points[include_id_list.index(ind)].append(point)
        masks, contact_forces = [], []
        for i, points in enumerate(contact_points):
            if len(points) > 0:
                contact_force = np.sum([np.asarray(p[contactNormalOnB]) * p[normalForce] for p in points], axis=0)
            else:
                contact_force = np.zeros(3, dtype=np.float32)
            contact_forces.append(contact_force[-1])
            masks.append(abs(contact_force[-1]) >= 1.)
        return {
            'mask': np.asarray(masks),
            'force': np.asarray(contact_forces),
        }

    def get_scanned_height_around_foot(self, bound: float = 0.15):
        yaw = self.aliengo.rpy[2]
        scanned_point_offsets = bound * np.asarray([(0., 0.)] + [(cos(theta), sin(theta)) for theta in np.arange(0, 2 * pi, pi / 2) + yaw])
        # scan_point_offsets = bound * np.asarray([(cos(yaw), sin(yaw))])
        scanned_heights = []
        for foot_pos in self.aliengo.foot_position:
            scanned_heights.append([foot_pos[2] - self.terrain.get_height(foot_pos[:2] + offset) for offset in scanned_point_offsets])
        return np.asarray(scanned_heights) - 0.0265

    def get_scanned_height_under_foot(self):
        scanned_heights = []
        for foot_pos in self.aliengo.foot_position:
            scanned_heights.append([foot_pos[2] - self.terrain.get_height(foot_pos[:2])])
        return np.asarray(scanned_heights) - 0.0265

    def setPhysicsEngineParameter(self):
        high_performance_mode = self.aliengo.velocity[2] < -2.
        if self._high_performance_mode != high_performance_mode:
            if high_performance_mode:
                self.client.setPhysicsEngineParameter(enableConeFriction=0, numSolverIterations=100, numSubSteps=2)
            else:
                self.client.setPhysicsEngineParameter(enableConeFriction=0, numSolverIterations=50, numSubSteps=0)
            self._high_performance_mode = high_performance_mode

    @property
    def terrain_penetration(self, threshold=0.01):
        foot_position = self.aliengo.foot_position
        foot_height = foot_position[:, 2]
        terrain_height = np.array([self.terrain.get_height(pos[:2]) for pos in foot_position])
        return np.any(foot_height - terrain_height <= -threshold)

    def get_external_force(self):
        return self.push_randomizer.force

    def get_physics_param(self):
        return self.param_randomizer.param

    def set_control_time_step(self, control_time_step):  # s
        self._action_repeat = int(round(control_time_step / self._time_step))
        self._control_time_step = self._time_step * self._action_repeat

    @property
    def hard_reset(self):
        return self._hard_reset

    @property
    def terrain_updated(self):
        return self.terrain_randomizer and self.terrain_randomizer.terrain_param_updated

    @property
    def control_time_step(self):
        return self._control_time_step

    @property
    def reset_count(self):
        return self._reset_count

    @property
    def counter(self):
        return self._counter

    @property
    def time(self):
        return self._counter * self._time_duration

    @property
    def is_render(self):
        return self._is_render

    @property
    def observation_name(self):
        return ('position', 'velocity', 'rpy', 'rpy_rate', 'motor_position', 'motor_velocity', 'motor_torque')

    @property
    def debug_name(self):
        motor_names = [name.strip('_joint') for name in self.aliengo.motor_names]
        return {
            'position': ['x', 'y', 'z'],
            'velocity': ['x', 'y', 'z'],
            'rpy': ['x', 'y', 'z'],
            'rpy_rate': ['x', 'y', 'z'],
            'motor_position': motor_names,
            'motor_velocity': motor_names,
            'motor_torque': motor_names,
            'action': motor_names,
        }

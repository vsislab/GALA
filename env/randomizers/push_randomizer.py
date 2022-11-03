"""Adds random forces to the base of aliengo during the simulation steps."""

from math import pi, sin, cos
import numpy as np
import pybullet_data

from .base_randomizer import BaseRandomizer

COLOR = np.array([1, 0, 1])  # pink


class PushRandomizer(BaseRandomizer):
    """Applies a random impulse to the base of aliengo."""

    def __init__(
            self,
            start_step=100,
            interval_step=500,
            duration_step=30,
            horizontal_force=20,
            vertical_force=5,
            # position=np.zeros(3),
            # position=np.array([0.08, 0.25, 0]),
            position=np.array([0.2, 0.2, 0]),
            push_strength_ratio=1.,
            render=False,
            **kwargs
    ):
        """Initializes the randomizer.

        Args:
          start_step: No push force before the env has advanced
            this amount of steps.
          interval_step: The step interval between applying
            push forces.
          duration_step: The duration of the push force.
          horizontal_force: The applied force magnitude when projected in the horizontal plane.
          vertical_force: The z component of the applied push force (positive:↓).
        """
        super(PushRandomizer, self).__init__(**kwargs)
        assert duration_step <= interval_step
        self._start_step = start_step
        self._interval_step = interval_step
        self._duration_step = duration_step
        self._horizontal_force = horizontal_force
        self._vertical_force = vertical_force
        self._position = position
        self.push_strength_ratio = push_strength_ratio
        self._is_render = render
        self._step = start_step
        self._link_id = 0
        self._random_force = np.zeros(3)
        self._force = np.zeros(3)

    def _randomize_env(self, env):
        theta = -pi / 2  # np.random.uniform(0, 2 * pi)
        # self._random_force = self._horizontal_force * np.array([cos(theta), sin(theta), 0]) \
        #                      + np.array([0, 0, -self._vertical_force])
        self._random_force = np.array([self._horizontal_force * cos(theta), self._horizontal_force * sin(theta), -np.random.choice([-1, 1]) * self._vertical_force])

    def _randomize_step(self, env):
        """Randomizes simulation steps.

        Will be called at every time step. May add random forces/torques to aliengo.

        Args:
          env: The aliengo gym environment to be randomized.
        """
        if env.counter >= self._step + self._interval_step:
            self._randomize_env(env)
            self._step = env.counter
        if self._step <= env.counter < self._step + self._duration_step:
            # if (env.counter >= self._start_step) and ((env.counter - self._start_step) % self._interval_step < self._duration_step):
            force = self._random_force * self.push_strength_ratio
            env.client.applyExternalForce(objectUniqueId=env.aliengo.urdf_id,
                                          linkIndex=self._link_id,
                                          forceObj=force,
                                          posObj=self._position,
                                          flags=env.client.LINK_FRAME)
            if self._is_render and env.is_render:
                applied_position = env.aliengo.get_link_position([1])[0] + self._position
                env.client.addUserDebugLine(applied_position,
                                            applied_position - force / 450,
                                            lineColorRGB=COLOR,
                                            lineWidth=4,
                                            lifeTime=0.005)
        else:
            force = np.zeros(3)
        self._force = force

    @property
    def link_id(self):
        return self._link_id

    @property
    def force(self):
        return self._force

    @property
    def position(self):
        return self._position


class HitRandomizer(BaseRandomizer):
    def __init__(self,
                 start_step=0,
                 interval_steps=300,
                 duration_steps=3,
                 **kwargs
                 ):
        super(HitRandomizer, self).__init__(**kwargs)
        self._start_step = start_step
        self._interval_steps = interval_steps
        self._duration_steps = duration_steps
        self.ball_num = 2
        self.balls = None
        self.count = 0

    def _randomize_env(self, env):
        if self.balls is None:
            self.balls = []
            for _ in range(self.ball_num):
                scaling = np.random.uniform(0.06, 0.12)
                # ball = env.client.loadURDF("%s/soccerball.urdf" % pybullet_data.getDataPath(), globalScaling=scaling)
                ball = env.client.loadURDF("%s/sphere2red.urdf" % pybullet_data.getDataPath(), globalScaling=scaling)
                env.client.changeDynamics(ball, -1, mass=scaling, linearDamping=0, angularDamping=0, rollingFriction=0.001, spinningFriction=0.001)
                # env.client.changeVisualShape(ball, -1, rgbaColor=[0.8, 0.8, 0.8, 1])
                self.balls.append(ball)

    def _randomize_step(self, env):
        if env.counter >= self._start_step and \
                env.counter % self._interval_steps < self._duration_steps * len(self.balls):
            if self.count % self._duration_steps == 0:
                ball = self.balls[self.count // self._duration_steps]
                r, theta, phi = 1., np.random.uniform(pi / 3, pi / 1.8), np.random.uniform(0, 2 * pi)
                x, y, z = r * sin(theta) * cos(phi), r * sin(theta) * sin(phi), r * cos(theta)
                pos = np.array([x, y, z])
                vel = -pos  # * 6
                env.client.resetBasePositionAndOrientation(ball,
                                                           pos + env.aliengo.position + np.random.uniform(-0.15, 0.15, 3),
                                                           [0, 0, 0, 1])
                env.client.resetBaseVelocity(ball, vel)
            self.count += 1
        else:
            self.count = 0


if __name__ == '__main__':
    import pybullet
    import pybullet_utils.bullet_client
    import pybullet_data
    import time


    class Env:
        def __init__(self):
            self.client = pybullet_utils.bullet_client.BulletClient(connection_mode=pybullet.GUI)
            self.client.resetSimulation()
            self.client.setTimeStep(0.01)
            self.client.setGravity(0, 0, -9.8)
            self.client.configureDebugVisualizer(self.client.COV_ENABLE_GUI, 0)
            # self.client.configureDebugVisualizer(self.client.COV_ENABLE_DEPTH_BUFFER_PREVIEW,0)
            # self.client.configureDebugVisualizer(self.client.COV_ENABLE_RGB_BUFFER_PREVIEW,0)
            # self.client.configureDebugVisualizer(self.client.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,0)
            self.terrain_id = self.client.loadURDF("%s/plane.urdf" % pybullet_data.getDataPath())
            self.counter = 0
            self.step()

        def step(self):
            self.client.stepSimulation()
            self.counter += 1


    env = Env()
    randomizer = HitRandomizer()

    env.client.configureDebugVisualizer(env.client.COV_ENABLE_RENDERING, 0)
    env.client.configureDebugVisualizer(env.client.COV_ENABLE_RENDERING, 1)
    randomizer.randomize_env(env)
    for i in range(3000):
        env.client.configureDebugVisualizer(env.client.COV_ENABLE_SINGLE_STEP_RENDERING, 1)  # smooth simulation rendering
        randomizer.randomize_step(env)
        env.step()

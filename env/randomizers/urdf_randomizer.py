# -*- coding: utf-8 -*-
import os
from urdfpy import URDF
import numpy as np
import warnings
from .base_randomizer import BaseRandomizer


class UrdfRandomizer(BaseRandomizer):
    com_range = (
        (0, -0.03),  # x
        (-0.01, 0.01),  # y
        (0, -0.02)  # z
    )

    def __init__(self, urdf_file, output_dir, epoch_interval=100, **kwargs):
        super(UrdfRandomizer, self).__init__(**kwargs)
        self._file_name = urdf_file.split('/')[-1]
        self.robot = URDF.load(urdf_file)
        self.output_dir = output_dir
        self._epoch_interval = epoch_interval

    def _randomize_step(self, env):
        pass

    def _randomize_env(self, env):
        if env.reset_count % self._epoch_interval == 0:
            self.random_com()
            urdf_file = os.path.join(self.output_dir, str(os.getpid()), self._file_name)
            self.robot.save(urdf_file)
            env.aliengo.urdf_file = urdf_file

    def random_com(self, com: tuple = None):
        if com is None:
            com = tuple(np.random.uniform(*range) for range in self.com_range)
        else:
            assert len(com) == 3
        self.robot.link_map['trunk_inertia'].inertial.origin[[0, 1, 2], -1] = com
        return com


if __name__ == '__main__':
    urdf_randomizer = UrdfRandomizer('env/assets/aliengo/urdf/aliengo.urdf', output_dir='tmp/urdf')


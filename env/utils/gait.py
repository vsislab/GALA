# -*- coding: utf-8 -*-
from math import pi
import numpy as np

from env.utils import IK, VerticalTrajectoryGenerator, SinTrajectoryGenerator


class TrotGait:
    def __init__(self, env):
        self.env = env
        self.ik = IK(env.aliengo.leg_length)
        self.height_range = (0.04, 0.08)
        self.frequency_range = (0.16, 4)
        self.stride_range = (0.005, 0.03)
        self.phi0_alternative = ((0, pi, pi, 0), (pi, 0, 0, pi))
        self.x_range = (-0.02, 0.02)
        self.reset()

    def reset(self):
        self.height = np.random.uniform(*self.height_range)
        self.frequency = np.random.uniform(*self.frequency_range)
        self.stride = np.random.uniform(*self.stride_range)
        self.x = np.random.uniform(*self.x_range)
        self.phi0 = self.phi0_alternative[np.random.choice((0, 1))]

        self.sin_tgs = [SinTrajectoryGenerator(self.env.control_time_step, self.stride, self.frequency, phi + pi / 2) for phi in self.phi0]
        self.vtgs = [VerticalTrajectoryGenerator(self.env.control_time_step, self.height, self.frequency, phi) for phi in self.phi0]

    def step(self):
        pos = np.stack([[self.x, self.sin_tgs[i].compute(), self.vtgs[i].compute()] for i in range(4)])
        pos = pos + self.env.aliengo.FOOT_POSITION_REFERENCE
        action = np.stack([self.ik.inverse_kinematics(*p) for p in pos]).transpose().flatten()
        return action  # + np.random.normal(size=action.size) * self._action_std


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import mpl_toolkits.axisartist as axisartist

    fig = plt.figure()
    ax = axisartist.Subplot(fig, 111)
    fig.add_axes(ax)
    ax.axis[:].set_visible(False)
    ax.axis["x"] = ax.new_floating_axis(0, 0, axis_direction="bottom")
    ax.axis["y"] = ax.new_floating_axis(1, 0, axis_direction="bottom")
    ax.axis["x"].set_axisline_style("->", size=1.0)
    ax.axis["y"].set_axisline_style("->", size=1.0)
    y_numbers = [-1, 1]
    y_labels = ["-1", "1"]
    plt.xticks(y_numbers, y_labels)

    sin_tg = SinTrajectoryGenerator(0.01, 0.1, 1, pi / 2)
    vtg = VerticalTrajectoryGenerator(0.01, 0.1, 1, 0)
    ys, zs = [], []
    phi = np.linspace(0.5 * pi, 2.5 * pi, 100)
    for _ in phi:
        ys.append(sin_tg.compute())
        zs.append(vtg.compute())
    plt.plot(phi, ys)
    plt.plot(phi, zs)
    plt.show()
    # plt.plot(ys, zs)
    # plt.show()

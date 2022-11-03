# -*- coding: utf-8 -*-
from math import sin, pi, tau
import numpy as np


class CycloidTrajectoryGenerator:
    def __init__(self, stride, height, period):
        self.stride = stride
        self.height = height
        self.period = period
        self._half_period = period / 2

    def _swing(self, t):
        x = self.stride * (t / self._half_period - 1 / tau * sin(tau * t / self._half_period))
        y = self.height * (np.sign(self._half_period / 2 - t) * (
                2 * (t / self._half_period - 1 / (4 * pi) * sin(4 * pi * t / self._half_period)) - 1) + 1)
        return x, y

    def _support(self, t):
        x = self.stride * (1 - t / self._half_period + 1 / tau * sin(tau * t / self._half_period))
        y = 0 * x
        return x, y

    def compute(self, t):
        t %= self.period
        if t <= self._half_period:
            return self._swing(t)
        else:
            return self._support(t - self._half_period)


class PhaseModulator:
    def __init__(self, time_step, f0=0.):
        self._time_step = time_step
        self._f0 = f0
        self.reset()

    def reset(self, phi0=0.):
        self._phi = self._phi0 = phi0 % tau

    def compute(self, f=0):
        phi = self._phi + tau * (self._f0 + f) * self._time_step
        self._phi = phi % tau
        return self._phi

    @property
    def phi(self):
        return self._phi

    @property
    def phi0(self):
        return self._phi0

    @property
    def f0(self):
        return self._f0


class BaseTrajectoryGenerator:
    def __init__(self, time_step, h0, f0, phi0=None):
        self._time_step = time_step
        self._h0 = h0
        self._f0 = f0
        self._init_phi0 = phi0 % tau
        self.reset()

    def reset(self, phi0=None):
        if phi0 is not None:
            self._phi0 = phi0
        elif self._init_phi0 is not None:
            self._phi0 = self._init_phi0
        else:
            self._phi0 = np.random.uniform(0, tau)
        self._phi = self._phi0
        self._f = 0.

    def compute(self, f=0, h=0):
        self._f = f
        self._h = h
        self._phi += tau * (self._f0 + f) * self._time_step
        self._phi = self._phi % tau
        return self._compute(self._phi, self._h0 + h)

    def _compute(self, f, h):
        raise NotImplementedError

    @property
    def phi0(self):
        return self._phi0

    @property
    def phi(self):
        return self._phi

    @property
    def f0(self):
        return self._f0

    @property
    def h0(self):
        return self._h0

    @property
    def f(self):
        return self._f

    @property
    def h(self):
        return self._h


class SinTrajectoryGenerator(BaseTrajectoryGenerator):
    def __init__(self, time_step, h0=0.1, f0=1.25, phi0=0):
        super(SinTrajectoryGenerator, self).__init__(time_step, h0, f0, phi0)

    def _compute(self, phi, h):
        return h * sin(phi)


class VerticalTrajectoryGenerator(BaseTrajectoryGenerator):
    def __init__(self, time_step, h0=0.1, f0=1.25, phi0=0.):
        super(VerticalTrajectoryGenerator, self).__init__(time_step, h0, f0, phi0)

    def _compute(self, phi, h):
        """-2 <= k <= 2"""
        k = 2 * (phi - pi) / pi
        if 0 <= k < 1:
            z = h * (-2 * k ** 3 + 3 * k ** 2)
        elif 1 <= k < 2:
            z = h * (2 * k ** 3 - 9 * k ** 2 + 12 * k - 4)
        else:
            z = 0
        return z


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    mode = 2

    if mode == 0:
        print('Test cycloid trajectory generator.')
        STRIDE, HEIGHT, PERIOD = 150, 100, 500  # m,m,step
        cycloid_TG = CycloidTrajectoryGenerator(STRIDE, HEIGHT, PERIOD)
        ts = np.arange(PERIOD)
        x, y = np.stack(cycloid_TG.compute(t) for t in ts).transpose()
        plt.plot(x, y, 'r')
        plt.show()

    # ------------------------------------------

    if mode == 1:
        print('Test sin trajectory generator.')
        sin_tg = SinTrajectoryGenerator(0.01, h0=0.1, f0=1, phi0=0)
        ts = np.arange(0, 1, 0.01)
        zs = np.array([sin_tg.compute() for t in ts])
        plt.plot(ts, zs)
        plt.show()

    # ------------------------------------------

    if mode == 2:
        print('Test vertical trajectory generator.')
        vtg = VerticalTrajectoryGenerator(0.01, h0=0.1, f0=1, phi0=0)
        ts = np.arange(0, 1, 0.01)
        zs = np.array([vtg.compute() for t in ts])
        plt.plot(ts, zs)
        plt.show()

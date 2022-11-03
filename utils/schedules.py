# -*- coding: utf-8 -*-
import math


class BaseSchedule:
    def __init__(self, init_value):
        self._init_value = init_value
        self.reset()

    def reset(self):
        self._value = self._init_value
        self._step = 0

    def step(self):
        raise NotImplementedError

    @property
    def value(self):
        return self._value


class ETHSchedule(BaseSchedule):
    def __init__(self, init_value, ratio):
        super(ETHSchedule, self).__init__(init_value)
        self._ratio = ratio

    def step(self):
        self._step += 1
        self._value = self._value ** self._ratio
        return self._value


class LinearSchedule(BaseSchedule):
    def __init__(self, init_value, final_value, start_step, duration_step):
        assert start_step >= 0 and duration_step > 0
        super(LinearSchedule, self).__init__(init_value)
        self._final_value = final_value
        self._start_step = start_step
        self._step_value = (final_value - init_value) / duration_step
        self._limit_func = min if self._step_value > 0 else max

    def step(self):
        self._step += 1
        if self._step >= self._start_step:
            self._value = self._limit_func(self._value + self._step_value, self._final_value)
        return self._value


class CosineSchedule(BaseSchedule):
    def __init__(self, init_value, final_value, duration_step):
        assert duration_step >= 1
        super(CosineSchedule, self).__init__(init_value)
        self._final_value = final_value
        self._duration_step = duration_step

    def step(self):
        if self._step == self._duration_step:
            self.reset()
        self._step += 1
        self._value = (math.cos(math.pi / self._duration_step * self._step) + 1) / 2 * (self._init_value - self._final_value) + self._final_value
        return self._value


class ConstantSchedule(BaseSchedule):
    def __init__(self, init_value):
        super(ConstantSchedule, self).__init__(init_value)

    def step(self):
        self._step += 1
        return self._value


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    schedule = ETHSchedule(0.3, 0.995)
    ts = np.arange(1500)
    ys = []
    for t in ts:
        schedule.step()
        ys.append(schedule.value)
    plt.plot(ts, ys)
    plt.show()

# -*- coding: utf-8 -*-
import numpy as np

TERRAIN_CALLBACK_PARAM = ['box_height', 'step_height', 'step_width', 'amplitude', 'roughness', 'slope']
PUSH_CALLBACK_PARAM = ['push_strength_ratio']


class CallbackParam:
    """
    params: key = (start, end, step)
    """

    def __init__(self, name: str, success: bool = False, **ranges):
        self.name = name
        self.success = success
        self._raw_ranges = ranges
        self._ranges = {}
        for key, value in ranges.items():
            assert len(value) == 3
            self.__dict__[key] = value[0]
            if value[0] <= value[1]:
                range = (value[0], value[1], abs(value[2]))
            else:
                range = (value[1], value[0], -abs(value[2]))
            self._ranges[key] = range
            # setattr(self, key, value[0])

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        assert key in self.__dict__
        min_value, max_value, _ = self._ranges[key]
        self.__dict__[key] = max(min_value, min(max_value, value))

    def update(self, reverse=False):
        if not self.success:
            self.success = np.all([self.__dict__[key] == self._raw_ranges[key][1] for key in self._ranges])
        for key in self._ranges:
            min_value, max_value, step_value = self._ranges[key]
            if self.success:
                # if key == 'box_height':
                #     self.__dict__[key] = self._raw_ranges[key][1]
                # else:
                self.__dict__[key] = np.random.uniform(min_value, max_value)
            else:
                if reverse:
                    step_value = -step_value
                self.__dict__[key] = max(min_value, min(max_value, self.__dict__[key] + step_value))

    @property
    def terrain_value(self):
        return {key: self.__dict__[key] for key in self._ranges if key in TERRAIN_CALLBACK_PARAM}

    @property
    def push_value(self):
        return {key: self.__dict__[key] for key in self._ranges if key in PUSH_CALLBACK_PARAM}

    @property
    def value(self):
        return {key: self.__dict__[key] for key in self._ranges}


if __name__ == '__main__':
    c = CallbackParam('Flat', step_height=(1, 0, 0.5))
    print(c.value)
    c.update()
    c.update()
    print(c.value)
    for _ in range(100):
        c.update(True)
    print(c.success)
    print(c.value)

# -*- coding: utf-8 -*-
import collections
import enum
import numpy as np


class FootStatus(enum.Enum):
    STARTING_POINT = 0
    SWING = 1
    DROP_POINT = 2
    SUPPORT = 3


class FootStatusChecker:
    def __init__(self, length):
        self._foot_contact_history = collections.deque(maxlen=length)
        self._status = None

    def reset(self, contact):
        """contact should be bool list"""
        self._foot_contact_history.clear()
        self._foot_contact_history.append(contact)
        self._status = [FootStatus.SUPPORT if c else FootStatus.SWING for c in contact]

    def check(self, contact):
        self._foot_contact_history.append(contact)
        foot_contact_history = np.stack(self._foot_contact_history)
        for i in range(foot_contact_history.shape[1]):
            c = foot_contact_history[:, i]
            if all(c):
                if self._status[i] is FootStatus.SWING:
                    self._status[i] = FootStatus.DROP_POINT
                elif self._status[i] is FootStatus.DROP_POINT:
                    self._status[i] = FootStatus.SUPPORT
                elif self._status[i] is FootStatus.STARTING_POINT:
                    self._status[i] = FootStatus.SUPPORT
            elif all(np.logical_not(c)):
                if self._status[i] is FootStatus.SUPPORT:
                    self._status[i] = FootStatus.STARTING_POINT
                elif self._status[i] is FootStatus.STARTING_POINT:
                    self._status[i] = FootStatus.SWING
                elif self._status[i] is FootStatus.DROP_POINT:
                    self._status[i] = FootStatus.SWING
        return self._status

    @property
    def status(self):
        return self._status


if __name__ == '__main__':
    x=[FootStatus.DROP_POINT,FootStatus.STARTING_POINT]
    print(FootStatus.SWING )

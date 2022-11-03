# -*- coding: utf-8 -*-
from math import sin, cos, tau
import numpy as np
import random
import pybullet as p
import torch


def _get_right_size_value(value, size):
    if np.isscalar(value):
        _size = size
    else:
        _size = int(size / len(value))
    # _size=size if np.isscalar(value) else int(size/len(value))
    value = np.repeat(value, _size)
    assert len(value) == size
    return value


def _get_right_value(value, default_value):
    return value if value is not None else default_value


def seed_all(seed=None):
    """
    Seed all devices deterministically off of seed and somewhat independently.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def smallest_signed_angle_between(x, y):
    """from X to Y"""
    a = (x - y) % tau
    b = (y - x) % tau
    return -a if a < b else b


def getZMatrixFromEuler(y):
    y_sin, y_cos = sin(y), cos(y)
    return np.asarray([
        [y_cos, -y_sin, 0.],
        [y_sin, y_cos, 0.],
        [0., 0., 1.],
    ])


def getMatrixFromEuler(rpy):
    """each row is the representation of old axis in new axis"""
    quaternion = p.getQuaternionFromEuler(rpy)
    matrix = p.getMatrixFromQuaternion(quaternion)
    return np.reshape(matrix, (3, 3), order='C')


CENTER_OFFSET = np.array([
    [0.196, -0.05, 0.],
    [0.196, 0.05, 0.],
    [- 0.196, -0.05, 0.],
    [- 0.196, 0.05, 0.]]) * 0


def convert_to_horizontal_frame(pos, rpy):
    """
    pos: base frame
    rpy: base rpy
    return: base frame
    """
    r, p, y = rpy
    pos = pos[:, [1, 0, 2]] * np.array([[1, -1, 1], [1, 1, 1], [1, -1, 1], [1, 1, 1]])  # base frame --> world frame
    return convert_world_to_base_frame(pos + CENTER_OFFSET, (r, p, 0))
    # r, p, y = rpy
    # pos = [np.dot(p, getMatrixFromEuler(o)) for o, p in zip([[-p, r, 0.], [-p, -r, 0.]], [pos[[0, 2]], pos[[1, 3]]])]
    # return np.concatenate(pos)[[0, 2, 1, 3]]


def convert_world_to_base_frame(pos, rpy):
    """
    pos: world frame
    rpy: base rpy
    return: base frame
    """
    pos = np.dot(pos, getMatrixFromEuler(rpy)) - CENTER_OFFSET
    pos = pos[:, [1, 0, 2]] * np.array([[-1, 1, 1], [1, 1, 1], [-1, 1, 1], [1, 1, 1]])  # world frame --> base frame
    return pos


def true_mask(start, stop, size):
    mask = np.zeros(size, dtype=bool)
    mask[start:stop] = True
    return mask


def fake_mask(start, stop, size):
    mask = np.ones(size, dtype=bool)
    mask[start:stop] = False
    return mask


if __name__ == '__main__':
    from math import pi, tau

    # from env import Aliengo
    #
    # # pose = np.array([
    # #     [0.1, 0.2, 0.3],
    # #     [0.11, 0.21, 0.31],
    # #     [0.1, 0.2, 0.3],
    # #     [0.11, 0.21, 0.31]
    # # ])
    # # rpy = np.array([-pi / 6, pi / 6, 0])
    # # print(pose)
    # # print(convert_to_horizontal_frame(pose, rpy))
    #
    # pos = np.stack([Aliengo.FOOT_POSITION_REFERENCE] * 4)
    # print(pos)
    # rpy = (pi / 8, pi / 8, 0)
    # res = convert_to_horizontal_frame(pos, rpy)
    # print(res)

    # vel = [1., 2, 3]
    # rpy = [-pi/5, pi / 2, pi/3]
    # print(np.dot(vel, getMatrixFromEuler(rpy)))

    # ang = smallest_signed_angle_between(-3, 3)
    # print(ang)

    rpy = [0.78, 0, 0]
    print(getMatrixFromEuler(rpy))

    vel = [0.361679, 0.00632179, -0.0131936]
    rpy = [-0.0176776, 0.121894, 0.0529793]
    x = np.dot(vel, getMatrixFromEuler(rpy))
    print(x)

# -*- coding: utf-8 -*-
from math import sin, cos, asin, acos, atan2, atan, pi
import numpy as np


# condition 1
# alpha: -pi/4 ~ pi/4
#  beta: -pi/4 ~ pi/4
# gamma:     0 ~ pi/2

# OR

# condition 2
# alpha: -pi/2 ~ pi/2
#  beta: -pi/2 ~ pi/2
# gamma:     0 ~ pi
# subject to
# x>0 and z<-a
# where x, y, z are foot position and
#       a, b, c are the lengths of the three limbs on the leg.


class IK:
    def __init__(self, leg_length):
        assert len(leg_length) == 3
        self.leg_length = np.array(leg_length)

    def kinematics(self, alpha, beta, gamma):
        a, b, c = self.leg_length
        x = a * cos(alpha) + c * (sin(alpha) * sin(beta) * sin(gamma) - sin(alpha) * cos(beta) * cos(gamma)) - b * sin(alpha) * cos(beta)
        y = b * sin(beta) + c * (cos(beta) * sin(gamma) + sin(beta) * cos(gamma))
        z = -a * sin(alpha) + c * (cos(alpha) * sin(beta) * sin(gamma) - cos(alpha) * cos(beta) * cos(gamma)) - b * cos(alpha) * cos(beta)
        return np.asarray([x, y, z])

    def inverse_kinematics(self, x, y, z):
        a, b, c = self.leg_length
        L = x ** 2 + z ** 2 - a ** 2
        alpha = atan2(a, L ** 0.5) - atan2(x, -z)
        gamma = acos(((z ** 2 - x ** 2) * cos(alpha) ** 2 + x ** 2 + y ** 2 + x * z * sin(2 * alpha) - b ** 2 - c ** 2) / (2 * b * c))
        M = (c ** 2 - b ** 2 - y ** 2 - L) / (2 * b)
        N = (y - ((y ** 2 - M ** 2 + L) ** 0.5)) / (M - L ** 0.5)
        beta = - 2 * atan(N)
        return np.array([alpha, -beta, -gamma])

    def twh_inverse_kinematics(self, x, y, z):
        """TWH"""
        a, b, c = self.leg_length
        z_ = -(x ** 2 + z ** 2 - a ** 2) ** 0.5
        N = (c ** 2 - b ** 2 - y ** 2 - z_ ** 2) / (2 * b)
        M = (y - ((y ** 2 - N ** 2 + z_ ** 2) ** 0.5)) / (z_ + N)
        beta = pi / 4 - 2 * atan(M)
        gamma = pi / 4 - beta + asin((z_ / c) + (b / c) * (1 - M ** 2) / (1 + M ** 2))
        if x > 0:
            alpha = atan(-z / x) - atan(-z_ / a)
        elif x == 0:
            alpha = pi / 2 - atan(-z_ / a)
        else:
            alpha = pi - atan(z / x) - atan(-z_ / a)
        return np.array([alpha, beta - pi / 4, gamma + pi / 2])

    # def inverse_kinematics(self, x, y, z):
    #     """SJP"""
    #     a, b, c = self.leg_length
    #     ma, mb, mc = x, -z, a
    #     alpha = atan2(mc, (ma ** 2 + mb ** 2 - mc ** 2) ** 0.5) - atan2(ma, mb)
    #     gamma = acos(((z ** 2 - x ** 2) * cos(alpha) ** 2 + x ** 2 + y ** 2 + x * z * sin(2 * alpha) - b ** 2 - c ** 2) / (2 * b * c))
    #     ma, mb, mc = b + c * cos(gamma), -c * sin(gamma), -z * cos(alpha) - x * sin(alpha)
    #     beta = atan2(mc, -(ma ** 2 + mb ** 2 - mc ** 2) ** 0.5) - atan2(ma, mb)
    #     return np.array([alpha, beta, gamma])


def _get_right_range(x, y):
    # x = 0.076*2
    # y = 0.35
    from math import sqrt
    r = sqrt(0.083 ** 2 + (0.25 + 0.25) ** 2)  # 0.425
    z = sqrt(r ** 2 - x ** 2 - y ** 2)
    return z


if __name__ == '__main__':
    a, b, c = leg_length = [0.083, 0.25, 0.25]
    ik = IK(leg_length)

    """Test condition 1"""
    # for i in range(10000):
    #     angle = np.random.rand(3) * pi / 2 + np.array([-pi / 4, -pi / 4, 0])
    #     pos = ik.kinematics(*angle)
    #     _angle = ik.inverse_kinematics(*pos)
    #     _pos = ik.kinematics(*_angle)
    #     if np.round(_angle - angle, 3).sum() != 0:
    #         print(f'{i}:' + str(angle) + '\t' + str(_angle) + '\t' + str(np.round(_angle - angle, 3)))

    """Test condition 2"""
    # for i in range(10000):
    #     angle = np.random.rand(3) * pi + np.array([-pi / 2, -pi / 2, 0])
    #     pos = ik.kinematics(*angle)
    #     if pos[0] > 0 and pos[-1] < -a:
    #         _angle = ik.inverse_kinematics(*pos)
    #         _pos = ik.kinematics(*_angle)
    #         if np.round(_angle - angle, 3).sum() != 0:
    #             print(f'{i}:' + str(angle) + '\t' + str(_angle) + '\t' + str(np.round(_angle - angle, 3)))

    # """Test condition 3"""
    # for i in range(10000):
    #     pos = np.random.uniform([0, -0.2, -0.34], [0.15, 0.2, -0.1])
    #     angle = ik.inverse_kinematics(*pos)
    #     _pos = ik.kinematics(*angle)
    #     if np.round(_pos - pos, 3).sum() != 0:
    #         print(f'{i}:' + str(pos) + '\t' + str(_pos) + '\t' + str(np.round(_pos - pos, 3)))
    #     # if not np.logical_and(np.array([-pi / 2, -pi / 2, 0]) < angle , angle < np.array([pi / 2, pi / 2, pi])).all():
    #     #     print(f'{i}:' + str(pos) + '\t' + str(angle))

    """Test condition 4"""
    for i in range(10):
        ang = [-1.2, 0.67, -1.3]
        pos = ik.kinematics(*ang)
        print(f'{i}:' + str(pos))

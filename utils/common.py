# -*- coding: utf-8 -*-
import datetime
import math
from math import *
import numpy as np
import os
from os.path import exists
import shutil


def get_unique_num():
    return str(datetime.datetime.now().strftime("%Y%m%d%H%M%S")) + str(np.random.randint(10, 100))


def get_print_time(t):
    h = int(t // 3600)
    m = int((t // 60) % 60)
    s = int(t % 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def clear_dir(dir):
    if exists(dir):
        shutil.rmtree(dir)
        os.makedirs(dir, exist_ok=True)


def norm(data, norm_a, norm_b):
    return (data - norm_a) / norm_b


def denorm(data, norm_a, norm_b):
    return data * norm_b + norm_a


def compute_gaussian_sigma(ent):
    return sqrt(exp(2 * ent) / (2 * pi * e))


def compute_multi_gaussian_sigma(ent, dim):
    return ((2 * ent) / (dim * log(2 * pi * e))) ** (1 / dim)


def compute_entropy(sigma, dim):
    return (0.5 + 0.5 * math.log(2 * pi) + math.log(sigma)) ** dim


if __name__ == '__main__':
    import time

    start = time.time()
    time.sleep(1)
    end = time.time()
    print(get_print_time(end - start))
    print(get_print_time(24 * 3600 + 24 * 60 + 24))
    print(compute_entropy(0.01, 1))
    print(compute_gaussian_sigma(0))
    print(compute_multi_gaussian_sigma(1, 12))

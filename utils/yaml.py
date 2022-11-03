# -*- coding: utf-8 -*-
import numpy as np
from typing import Dict
import yaml

from .attrdict import AttrDict


def _parse_param(param: dict):
    _param = {}
    for k, v in param.items():
        if isinstance(v, AttrDict):
            v = _parse_param(v.to_dict())
        elif isinstance(v, dict):
            v = _parse_param(v)
        elif isinstance(v, np.ndarray):
            v = v.tolist()
        elif isinstance(v, tuple):
            v = list(v)
        _param[k] = v
    return _param


def write_param(file, param: Dict):
    with open(file, 'w', encoding='utf-8') as f:
        yaml.dump(_parse_param(param), f)


def read_param(file) -> Dict:
    with open(file, 'r', encoding='utf-8') as f:
        param = yaml.load(f, Loader=yaml.FullLoader)
    return param


def update_param(file, param: Dict) -> Dict:
    _param = read_param(file)
    _param.update(param)
    write_param(file, _param)
    return _param

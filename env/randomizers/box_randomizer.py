# -*- coding: utf-8 -*-
import enum
import math
import numpy as np
from os.path import dirname, join
from typing import List, Dict

from .base_randomizer import BaseRandomizer


class BoxType(enum.Enum):
    """The randomized box types we can use in the gym env."""
    Flat = 0
    Slope = 1
    Stair = 2


class BoxParam:
    def __init__(self, type: BoxType = BoxType.Flat, size: tuple = (2., 2.), **specials):
        self.type = type
        self.size = size
        self.specials = specials

    @classmethod
    def Template(cls, type: str = 'Flat', **specials):
        # elif type == 'Random':
        #     return BoxParam(type=BoxType.Random, size=(30., 30.), step_height=specials.get('step_height', 0.05))
        if type == 'Flat':
            return BoxParam(BoxType.Flat, size=(2, 2),
                            jump_height=specials.get('jump_height', 0.2))
        elif type == 'Slope':
            return BoxParam(BoxType.Slope, size=(2, 2),
                            jump_height=specials.get('jump_height', 0.2))
        elif type == 'Stair':
            return BoxParam(BoxType.Stair, size=(2, 2),
                            jump_height=specials.get('jump_height', 0.2))
        else:
            raise NameError(f'Not exist terrain {type}')


class BoxInstance:
    def __init__(self, client, ids: List[int], param: BoxParam):
        self.client = client
        self.ids = ids
        self.param = param
        self.base_positions = [np.asarray(client.getBasePositionAndOrientation(id)[0]) for id in ids]
        self._position = np.zeros(3, dtype=float)
        info = client.getDynamicsInfo(ids[0], -1)
        self._friction, self._restitution = info[1], info[5]

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, v):
        v = np.asarray(v)
        self._position = v
        for i, id in enumerate(self.ids):
            _, orientation = self.client.getBasePositionAndOrientation(id)
            self.client.resetBasePositionAndOrientation(id, self.base_positions[i] + v, orientation)

    @property
    def friction(self):
        return self._friction

    @friction.setter
    def friction(self, v):
        self._friction = v
        self.client.changeDynamics(self.ids[0], -1, lateralFriction=v)

    @property
    def restitution(self):
        return self._restitution

    @restitution.setter
    def restitution(self, v):
        self._restitution = v
        self.client.changeDynamics(self.ids[0], -1, restitution=v)


class BoxRandomizer(BaseRandomizer):
    assets_dir = join(dirname(__file__), '../assets')

    def __init__(self, box_param: BoxParam = BoxParam()):
        super(BoxRandomizer, self).__init__()
        self.box_randomizer_dict = {
            BoxType.Flat: self._generate_flat_platform,
            BoxType.Slope: self._generate_slope_platform,
            BoxType.Stair: self._generate_stair_platform,
        }
        self._box_param = box_param
        self.box_param_updated = True
        self.box: BoxInstance

    def _randomize_step(self, env):
        pass

    def _randomize_env(self, env):
        if self.box_param_updated:
            self.box_param_updated = False
            box_param = self._box_param
            box_ids = self.box_randomizer_dict[box_param.type](env.client, box_param.size, **box_param.specials)
            self.box = BoxInstance(env.client, box_ids, param=box_param)

    def _generate_flat_platform(self, client, size: tuple, jump_height: float) -> List:
        box_id = generate_box_object(client, np.zeros(3), size, jump_height)
        return [box_id]

    def _generate_slope_platform(self, client, size: tuple, jump_height: float) -> List:
        box_id = generate_box_object(client, np.zeros(3), size, jump_height)
        slope_id = generate_slope_object(client, np.array([size[0], 0, 0]), size, slope=15)
        return [box_id, slope_id]

    def _generate_stair_platform(self, client, size: tuple, jump_height: float) -> List:
        box_id = generate_box_object(client, np.zeros(3), size, jump_height)
        stair_ids = generate_stair_object(client, np.array([size[0], 0, 0]), size, step_width=0.3, step_height=0.05)
        return [box_id] + stair_ids

    @property
    def box_param(self):
        return self._box_param

    @box_param.setter
    def box_param(self, box_specials: Dict):
        self.box_param_updated = True
        for k, v in box_specials.items():
            self._box_param.specials[k] = v


def generate_box_object(client, position: np.ndarray, size: tuple, height=0.5) -> int:
    x_size, y_size = size
    shape_id = client.createCollisionShape(client.GEOM_BOX, halfExtents=np.asarray([x_size, y_size, height]) / 2)
    # texture_id = client.loadTexture("%s/box/grey.jpg" % self.assets_dir)
    box_id = client.createMultiBody(baseMass=0,
                                    baseCollisionShapeIndex=shape_id,
                                    basePosition=position + np.asarray([x_size, 0, height]) / 2,
                                    baseOrientation=[0.0, 0.0, 0.0, 1])
    # client.changeDynamics(box_id, -1, lateralFriction=0.8)
    # client.changeVisualShape(box_id, -1, rgbaColor=[1, 1, 1, 1], textureUniqueId=texture_id)
    return box_id


def generate_slope_object(client, position: np.ndarray, size: tuple, slope: float = 15) -> int:
    x_size, y_size = size
    slope = slope / 57.3
    shape_id = client.createCollisionShape(client.GEOM_BOX, halfExtents=np.asarray([x_size, y_size, 0.1]) / 2)
    # texture_id = client.loadTexture("%s/box/grey.jpg" % self.assets_dir)
    box_id = client.createMultiBody(baseMass=0,
                                    baseCollisionShapeIndex=shape_id,
                                    basePosition=position + np.asarray([math.cos(slope), 0, math.sin(slope)]) * x_size / 2,
                                    baseOrientation=client.getQuaternionFromEuler((0, -slope, 0)))
    # client.changeDynamics(box_id, -1, lateralFriction=0.8)
    # client.changeVisualShape(box_id, -1, rgbaColor=[1, 1, 1, 1], textureUniqueId=texture_id)
    return box_id


def generate_stair_object(client, position: np.ndarray, size: tuple, step_width=0.1, step_height=0.05) -> List:
    x_size, y_size = size
    shape_id = client.createCollisionShape(client.GEOM_BOX, halfExtents=np.asarray([step_width, y_size, step_height]) / 2)
    # texture_id = client.loadTexture("%s/box/grey.jpg" % self.assets_dir)
    box_ids = []
    for i in range(int(x_size / step_width) + 1):
        box_id = client.createMultiBody(baseMass=0,
                                        baseCollisionShapeIndex=shape_id,
                                        basePosition=position + np.asarray([step_width / 2 + i * step_width, 0, step_height / 2 + i * step_height]),
                                        baseOrientation=[0.0, 0.0, 0.0, 1])
        box_ids.append(box_id)
        # client.changeDynamics(box_id, -1, lateralFriction=0.8)
        # client.changeVisualShape(box_id, -1, rgbaColor=[1, 1, 1, 1], textureUniqueId=texture_id)
    return box_ids

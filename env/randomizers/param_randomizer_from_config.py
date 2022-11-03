"""An environment randomizer that randomizes physical parameters from config."""
from copy import deepcopy
from functools import partial
import numpy as np
from os.path import dirname, join
import yaml

from .base_randomizer import BaseRandomizer


class ParamRandomizerFromConfig(BaseRandomizer):
    """A randomizer that change the aliengo_gym_env during every reset."""

    def __init__(self, config: dict = None, **kwargs):
        super(ParamRandomizerFromConfig, self).__init__(**kwargs)
        with open(join(dirname(__file__), 'default_config.yaml'), 'r') as f:
            DEFAULT_CONFIG = yaml.load(f, Loader=yaml.FullLoader)
        self._param_range = DEFAULT_CONFIG if config is None else {**DEFAULT_CONFIG, **config}
        self._default_param = {}
        self._param = {}

    def init_param(self, param: dict):
        self._default_param = param
        self._param = param

    def _randomize_step(self, env):
        pass

    def _randomize_env(self, env):
        param_range = deepcopy(self._param_range)
        if env.terrain.param is not None:
            for key in ['friction', 'restitution']:
                name = f'{env.terrain.param.type.name} {key}'
                if name in param_range:
                    param_range[f'terrain {key}'] = param_range[name]
        randomization_function_dict = self._build_randomization_function_dict(env)
        for name in randomization_function_dict:
            if name in param_range:
                self._param[name] = randomization_function_dict[name](lower_bound=param_range[name][0], upper_bound=param_range[name][1])

    def _build_randomization_function_dict(self, env):
        func_dict = {}
        # -----------------The following ranges are in percentage-------------------
        func_dict["mass"] = partial(self._randomize_mass, aliengo=env.aliengo)
        func_dict["inertia"] = partial(self._randomize_inertia, aliengo=env.aliengo)
        # ------------------The following ranges are the physical values in SI unit--------------------
        func_dict["payload"] = partial(self._randomize_payload, aliengo=env.aliengo)
        func_dict["latency"] = partial(self._randomize_latency, aliengo=env.aliengo)
        func_dict["control time step"] = partial(self._randomize_control_time_step, env=env)
        func_dict["motor damping"] = partial(self._randomize_motor_damping, aliengo=env.aliengo)
        func_dict["motor friction"] = partial(self._randomize_motor_friction, aliengo=env.aliengo)
        func_dict["motor strength"] = partial(self._randomize_motor_strength, aliengo=env.aliengo)
        func_dict["terrain friction"] = partial(self._randomize_terrain_friction, env=env)
        func_dict["terrain restitution"] = partial(self._randomize_terrain_restitution, env=env)
        func_dict["KP"] = partial(self._randomize_motor_KP, motor=env.aliengo.motor)
        func_dict["KD"] = partial(self._randomize_motor_KD, motor=env.aliengo.motor)
        return func_dict

    # -----------------The following ranges are in percentage-------------------
    def _randomize_mass(self, aliengo, lower_bound, upper_bound):
        mass = aliengo.get_mass_from_urdf()
        random_ratio = np.random.uniform(lower_bound, upper_bound, len(mass))
        random_quantity = np.concatenate([r * m for r, m in zip(random_ratio, mass)], axis=0)
        aliengo.set_mass(random_quantity)
        return random_quantity

    def _randomize_inertia(self, aliengo, lower_bound, upper_bound):
        inertia = aliengo.get_inertia_from_urdf()
        random_ratio = np.random.uniform(lower_bound, upper_bound, (len(inertia), 1))
        random_quantity = np.concatenate([r * i for r, i in zip(random_ratio, inertia)], axis=0)
        aliengo.set_inertia(random_quantity)
        return random_quantity

    # ------------------The following ranges are the physical values in SI unit--------------------
    def _randomize_payload(self, aliengo, lower_bound, upper_bound):
        random_quantity = np.random.uniform(lower_bound, upper_bound)
        aliengo.set_payload(random_quantity)
        return random_quantity

    def _randomize_latency(self, aliengo, lower_bound, upper_bound):
        random_quantity = np.random.uniform(lower_bound, upper_bound)
        aliengo.set_latency(random_quantity)
        return random_quantity

    def _randomize_control_time_step(self, env, lower_bound, upper_bound):
        random_quantity = np.random.uniform(lower_bound, upper_bound)
        env.set_control_time_step(random_quantity)
        return random_quantity

    def _randomize_motor_damping(self, aliengo, lower_bound, upper_bound):
        random_quantity = np.random.uniform(lower_bound, upper_bound)
        aliengo.set_motor_damping(random_quantity)
        return random_quantity

    def _randomize_motor_friction(self, aliengo, lower_bound, upper_bound):
        random_quantity = np.random.uniform(lower_bound, upper_bound)
        aliengo.set_motor_friction(random_quantity)
        return random_quantity

    def _randomize_motor_strength(self, aliengo, lower_bound, upper_bound):
        random_quantity = np.random.uniform(lower_bound, upper_bound)
        aliengo.set_motor_strength_ratio(random_quantity)
        return random_quantity

    def _randomize_terrain_friction(self, env, lower_bound, upper_bound):
        random_quantity = np.random.uniform(lower_bound, upper_bound)
        env.terrain.friction = random_quantity
        return random_quantity

    def _randomize_terrain_restitution(self, env, lower_bound, upper_bound):
        random_quantity = np.random.uniform(lower_bound, upper_bound)
        env.terrain.restitution = random_quantity
        return random_quantity

    def _randomize_motor_KP(self, motor, lower_bound, upper_bound):
        random_quantity = np.random.uniform(lower_bound, upper_bound, 12)
        motor.set_KP(random_quantity)
        return random_quantity

    def _randomize_motor_KD(self, motor, lower_bound, upper_bound):
        random_quantity = np.random.uniform(lower_bound, upper_bound, 12)
        motor.set_KD(random_quantity)
        return random_quantity

    # def _randomize_physics_quantity(self, name, fn, lower_bound, upper_bound, dimension=None):
    #     random_quantity = np.random.uniform(lower_bound, upper_bound, dimension)
    #     fn(random_quantity)
    #     return random_quantity

    @property
    def param_range(self):
        return self._param_range

    @property
    def param(self):
        return self._param

    @property
    def default_param(self):
        return self._default_param


if __name__ == '__main__':
    env_randomizer = ParamRandomizerFromConfig()
    for param_name, random_range in env_randomizer.param.items():
        print(param_name, random_range)

"""Abstract base class for environment randomizer."""

import abc


class BaseRandomizer(metaclass=abc.ABCMeta):
    """Abstract base class for environment randomizer.

    Randomizes physical parameters of the objects in the simulation and adds
    perturbations to the stepping of the simulation.
    """

    def __init__(self, enabled=True):
        self._enabled = enabled

    def randomize_env(self, env):
        """Randomize the simulated_objects in the environment.

        Will be called at when env is reset. The physical parameters will be fixed
        for that episode and be randomized again in the next environment.reset().

        Args:
          env: The gym environment to be randomized.
        """
        if self._enabled:
            self._randomize_env(env)

    @abc.abstractmethod
    def _randomize_env(self, env):
        pass

    def randomize_step(self, env):
        """Randomize simulation steps.

        Will be called at every timestep. May add random forces/torques to Minitaur.

        Args:
          env: The gym environment to be randomized.
        """
        if self._enabled:
            self._randomize_step(env)

    @abc.abstractmethod
    def _randomize_step(self, env):
        pass

    def enable(self, mode=True):
        self._enabled = mode

    @property
    def enabled(self):
        return self._enabled

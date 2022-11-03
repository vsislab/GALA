import numpy as np
from typing import Union


class RunningMeanStd:
    """Calulates the running mean and std of a data stream.

    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """

    def __init__(self, mean: Union[float, np.ndarray] = 0.0, std: Union[float, np.ndarray] = 1.0) -> None:
        self.mean, self.var = mean, std
        self.count = 0

    def update(self, x: np.ndarray) -> None:
        """Add a batch of item into RMS with the same shape, modify mean/var/count."""
        batch_mean, batch_var = np.mean(x, axis=0), np.var(x, axis=0)
        batch_count = len(x)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        new_var = m_2 / total_count

        self.mean, self.var = new_mean, new_var
        self.count = total_count

    def get_mean(self):
        return self.mean

    def get_var(self):
        return self.var


class RunningMeanStdGroup:
    def __init__(self, size: int, mean: Union[float, np.ndarray] = 0.0, std: Union[float, np.ndarray] = 1.0) -> None:
        self.size = size
        self.group = {i: RunningMeanStd(mean, std) for i in range(size)}

    def update(self, x: np.ndarray, idx: np.ndarray):
        for i in range(self.size):
            mask = idx == i
            if any(mask):
                self.group[i].update(x[mask])

    def get_mean(self, idx: np.ndarray):
        mean = np.empty(len(idx))
        for i in range(self.size):
            mean[idx == i] = self.group[i].mean
        return mean

    def get_var(self, idx: np.ndarray):
        var = np.empty(len(idx))
        for i in range(self.size):
            var[idx == i] = self.group[i].var
        return var

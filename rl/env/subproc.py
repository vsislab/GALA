# -*- coding: utf-8 -*-
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import numpy as np
from typing import List, Dict, Callable, Optional, Union

from env import AliengoGymEnvWrapper
from model.agent import BaseAgent
from rl.data import Batch
from utils.common import get_unique_num
from .dummy import ReplayBuffer


def dict_stack(l: List[Dict]):
    if len(l) == 0: return {}
    return {key: np.stack([x[key] for x in l]) for key in l[0].keys()}


# def dict_cat(insts: List[Dict]):
#     if len(insts) == 0: return {}
#     return {key: np.concatenate([d[key] for d in insts]) for key in insts[0].keys()}


class SharedBufferCell:
    def __init__(self, name1: str, name: str, dtype: np.dtype, dim: int, max_size: int, create: bool = False):
        nbytes = np.empty((max_size, dim), dtype).nbytes
        self.shm = SharedMemory(name=name, create=create, size=nbytes)
        # shape = (max_size, dim) if (dim > 1 or name1 == 'rew') else (max_size,)
        shape = (max_size, dim) if dim > 1 else (max_size,)
        self.array = np.ndarray(shape=shape, dtype=dtype, buffer=self.shm.buf)
        self.create = create

    def store(self, start_index: int, value: np.ndarray):
        self.array[start_index:start_index + len(value)] = value[:]

    @property
    def data(self):
        return self.array

    def __del__(self):
        del self.array
        self.shm.close()
        if self.create:
            self.shm.unlink()


class SharedBuffer:
    def __init__(self, max_size: int, format_dict: dict, create: bool = False):
        self.cells = {}
        for name, format in format_dict.items():
            if isinstance(format, dict):
                self.cells[name] = SharedBuffer(max_size, format, create)
            else:
                shm_name, dtype, dim = format
                self.cells[name] = SharedBufferCell(name, shm_name, dtype, dim, max_size, create)
        self._max_size = max_size
        self._format_dict = format_dict

    def store(self, start_index: int, **kwargs):
        for name, value in kwargs.items():
            if isinstance(value, dict):
                self.cells[name].store(start_index, **value)
            else:
                self.cells[name].store(start_index, value)

    def __len__(self) -> int:
        return self._max_size

    @property
    def data(self):
        return {name: cell.data for name, cell in self.cells.items()}

    @property
    def max_size(self):
        return self._max_size

    @property
    def format_dict(self):
        return self._format_dict

    @staticmethod
    def create_format_dict(inst_dict: dict, prefix: str = None):
        if prefix is None:
            prefix = get_unique_num()
        format_dict = {}
        for name, inst in inst_dict.items():
            shm_name = f'{prefix}.{name}'
            if isinstance(inst, dict):
                format_dict[name] = SharedBuffer.create_format_dict(inst, shm_name)
            else:
                inst = np.asarray(inst)
                format_dict[name] = (shm_name, inst.dtype, inst.size)
        return format_dict


def _worker(
        parent: mp.connection.Connection,
        p: mp.connection.Connection,
        env_fn: Callable[[], AliengoGymEnvWrapper],
        agent: BaseAgent,
        buffer_fn: Callable[[], SharedBuffer],
        buffer_counter: mp.Value,
        step_counter: mp.Value,
        traffic_signal: mp.Value,
):
    parent.close()
    env = env_fn()
    global_buffer = buffer_fn()
    local_buffer = ReplayBuffer()

    def store_data():
        with buffer_counter.get_lock():
            start_index = buffer_counter.value
            buffer_counter.value = start_index + len(local_buffer)
        global_buffer.store(start_index, **local_buffer.data)
        local_buffer.reset()

    local_buffer.reset()
    obs = env.reset()
    agent.reset(obs)
    episode_step, episode_rew = 0, 0
    try:
        while True:
            try:
                cmd, data = p.recv()
            except EOFError:  # the pipe has been closed
                p.close()
                break
            if cmd == "collect":
                n_step = data
                metrics = []
                collected_episode, collected_step = 0, 0
                while True:
                    act = agent(obs)
                    obs_next, rew, done, info = env.step(act)
                    local_buffer.add(obs, act, rew, obs_next, done, info)
                    # if env.is_multi_critic_task:
                    #     rew = rew[0]
                    episode_rew += rew
                    episode_step += 1
                    collected_step += 1
                    if done:
                        collected_episode += 1
                        metrics.append({
                            'len': episode_step,
                            'rew': episode_rew / episode_step * 100,
                            # 'suc': info['success']
                        })
                        store_data()
                        obs = env.reset()
                        agent.reset(obs)
                        episode_step, episode_rew = 0, 0
                    else:
                        obs = obs_next
                    if collected_step % 100 == 0:
                        with step_counter.get_lock():
                            step_counter.value += 100
                        if step_counter.value > n_step:
                            with traffic_signal.get_lock():
                                traffic_signal.value = False
                    if not traffic_signal.value:
                        if len(local_buffer):
                            local_buffer.meta['done'][-1] = True
                            store_data()
                        break
                p.send({
                    'episode': collected_episode,
                    'step': collected_step,
                    **{name: np.mean(value) for name, value in dict_stack(metrics).items()}
                })
            elif cmd == "seed":
                p.send(env.seed(data) if hasattr(env, "seed") else None)
            elif cmd == "getattr":
                p.send(getattr(env, data) if hasattr(env, data) else None)
            elif cmd == "callback":
                res = env.callback(*data)
                p.send(res)
            elif cmd == "close":
                p.send(env.close())
                p.close()
                break
            else:
                p.close()
                raise NotImplementedError
    except KeyboardInterrupt:
        p.close()


class SubprocEnv:
    def __init__(self, env_fns: List[Callable[[], AliengoGymEnvWrapper]], agent: BaseAgent, buffer_size: int):
        self.env_num = len(env_fns)
        agent.to_device('cpu')
        agent.share_memory()

        tmp_env = env_fns[0]()
        obs = tmp_env.reset()
        agent.reset(obs)
        act = agent(obs)
        obs_next, rew, done, info = tmp_env.step(act)
        format_example = {'obs': obs, 'act': act, 'rew': rew, 'obs_next': obs_next, 'done': done, 'info': info}
        format_dict = SharedBuffer.create_format_dict(format_example)
        self.buffer = SharedBuffer(buffer_size, format_dict, create=True)
        buffer_fn = lambda: SharedBuffer(buffer_size, format_dict, create=False)
        tmp_env.close()
        del tmp_env

        self.buffer_counter = mp.Value('i', 0, lock=True)
        self.step_counter = mp.Value('i', 0, lock=True)
        self.traffic_signal = mp.Value('b', True, lock=True)
        self.reset_buffer()

        self.env_pipes, self.env_processes = [], []
        for i in range(self.env_num):
            parent_remote, child_remote = mp.Pipe()
            args = (parent_remote, child_remote, env_fns[i], agent, buffer_fn, self.buffer_counter, self.step_counter, self.traffic_signal)
            process = mp.Process(target=_worker, args=args, daemon=True)
            process.start()
            child_remote.close()
            self.env_pipes.append(parent_remote)
            self.env_processes.append(process)

    def reset_buffer(self):
        with self.buffer_counter.get_lock():
            self.buffer_counter.value = 0
        with self.step_counter.get_lock():
            self.step_counter.value = 0
        with self.traffic_signal.get_lock():
            self.traffic_signal.value = True

    @property
    def data(self) -> Batch:
        with self.buffer_counter.get_lock():
            size = self.buffer_counter.value
        return Batch(self.buffer.data)[:size]

    def collect(self, n_step: int):
        assert n_step < self.buffer.max_size, 'The allocated shared buffer is too small!!!'
        [p.send(["collect", n_step]) for p in self.env_pipes]
        return [p.recv() for p in self.env_pipes]

    def seed(self, seed: Optional[Union[int, List[int]]] = None):
        seed_list: Union[List[None], List[int]]
        if seed is None:
            seed_list = [seed] * self.env_num
        elif isinstance(seed, int):
            seed_list = [seed + i for i in range(self.env_num)]
        else:
            seed_list = seed
        [p.send(["seed", s]) for s, p in zip(seed_list, self.env_pipes)]
        return [p.recv() for p in self.env_pipes]

    def callback(self, func_name: str, params: dict = None, id: Optional[Union[int, List[int], np.ndarray]] = None):
        id = self._wrap_id(id)
        [self.env_pipes[i].send(["callback", (func_name, params)]) for i in id]
        return [self.env_pipes[i].recv() for i in id]

    def close(self):
        try:
            [p.send(["close", None]) for p in self.env_pipes]
            # mp may be deleted so it may raise AttributeError
            [p.recv() for p in self.env_pipes]
            [p.join() for p in self.env_processes]
        except (BrokenPipeError, EOFError, AttributeError):
            pass
        # ensure the subproc is terminated
        [p.terminate() for p in self.env_processes]

    def _wrap_id(self, id: Optional[Union[int, List[int], np.ndarray]] = None) -> Union[List[int], np.ndarray]:
        if id is None:
            id = list(range(self.env_num))
        elif np.isscalar(id):
            id = [id]
        return id

    def __getattr__(self, key: str):
        [p.send(["getattr", key]) for p in self.env_pipes]
        return [p.recv() for p in self.env_pipes]

    def __len__(self):
        return self.env_num

    def __del__(self):
        self.close()

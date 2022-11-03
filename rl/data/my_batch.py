# -*- coding: utf-8 -*-
import numpy as np
import pprint
from typing import Any, Union, Iterator


def _parse_value(v: Any) -> Union[np.ndarray, "Batch"]:
    if isinstance(v, np.ndarray) or isinstance(v, Batch):
        return v
    elif isinstance(v, dict):
        return Batch(v)
    else:
        raise TypeError('Only np.ndarray, dict, Batch are supported.')


class Batch:
    def __init__(self, batch_dict: Union[dict, "Batch"] = None, **kwargs):
        if batch_dict is not None:
            for k, v in batch_dict.items():
                self.__dict__[k] = _parse_value(v)
        if len(kwargs) > 0:
            self.__init__(kwargs)

    def __setattr__(self, key: str, value: np.ndarray) -> None:
        self.__dict__[key] = value

    def __getattr__(self, key: str) -> Any:
        return getattr(self.__dict__, key)

    def __getitem__(self, index: Union[str, int, slice, np.ndarray]) -> "Batch":
        if isinstance(index, str):
            return self.__dict__[index]
        return Batch({key: value[index] for key, value in self.items()})

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__

    def __len__(self):
        keys = list(self.keys())
        if len(keys) == 0:
            return 0
        return len(self[keys[0]])

    def __repr__(self) -> str:
        """Return str(self)."""
        s = self.__class__.__name__ + "(\n"
        flag = False
        for k, v in self.__dict__.items():
            rpl = "\n" + " " * (6 + len(k))
            obj = pprint.pformat(v).replace("\n", rpl)
            s += f"    {k}: {obj},\n"
            flag = True
        if flag:
            s += ")"
        else:
            s = self.__class__.__name__ + "()"
        return s

    def cat(self, batch_dict: Union[dict, "Batch"]):
        for k in self.keys():
            inst = batch_dict[k]
            if isinstance(inst, np.ndarray):
                self.__dict__[k] = np.concatenate([self.__dict__[k], inst])
            elif isinstance(inst, dict) or isinstance(inst, Batch):
                self.__dict__[k] = self.__dict__[k].cat(inst)
            else:
                raise TypeError('Only np.ndarray, dict, Batch are supported.')
        return self


if __name__ == '__main__':
    print(len(Batch()))
    # b=Batch({})
    # print(len(b))
    obs = np.arange(10).repeat(10).reshape((10, 10))
    act = np.arange(10)
    batch1 = Batch({'obs': obs, 'act': act})
    batch2 = Batch({'obs': obs, 'act': act})
    b = batch1.cat(batch2)
    pass

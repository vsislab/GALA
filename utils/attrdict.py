# -*- coding: utf-8 -*-
import pprint
from typing import Union, Any, Dict


class AttrDict:
    def __init__(self, d: Dict = None, **kwargs):
        if d is not None:
            for k, v in d.items():
                if isinstance(v, dict):
                    self.__dict__[k] = AttrDict(v)
                else:
                    self.__dict__[k] = v
        if len(kwargs) > 0:
            self.__init__(kwargs)

    def __setattr__(self, key: str, value: Any) -> None:
        self.__dict__[key] = value

    def __getattr__(self, key: str) -> Any:
        return getattr(self.__dict__, key, None)

    def __setitem__(self, key: str, value: Any):
        self.__dict__[key] = value

    def __getitem__(self, key: str):
        return self.__dict__[key]

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__

    def update(self, d: Union[Dict, "AttrDict"]) -> None:
        for k, v in d.items():
            if k in self and (isinstance(v, dict) or isinstance(v, AttrDict)):
                self[k].update(v)
            else:
                if isinstance(v, dict):
                    v = AttrDict(v)
                self[k] = v

    def to_dict(self) -> Dict:
        d = {}
        for k, v in self.items():
            if isinstance(v, AttrDict):
                v = v.to_dict()
            d[k] = v
        return d

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


if __name__ == '__main__':
    d = AttrDict({'m': AttrDict({'x': 1, 'y': 2})})
    d.update({
        'm': {'x': [1, ], 'z': 3}
    })
    print(d)

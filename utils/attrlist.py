# -*- coding: utf-8 -*-
import pprint
from typing import Union, Any, Dict, List, Tuple


class AttrList:
    def __init__(self, l: Union[List, Tuple]):
        for i, k in enumerate(l):
            self.__dict__[k] = i

    def __setattr__(self, key: str, value: Any) -> None:
        self.__dict__[key] = value

    def __getattr__(self, key: str) -> Any:
        return getattr(self.__dict__, key, None)

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__

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

if __name__=='__main__':
    l=AttrList(['a','b','c'])
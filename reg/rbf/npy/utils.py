import numpy as np
import torch

from functools import wraps


def transform(arr, trans=None):
    if trans is None:
        return arr
    else:
        return trans.transform(arr)


def inverse_transform(arr, trans=None):
    if trans is None:
        return arr
    else:
        return trans.inverse_transform(arr)


def atleast_2d(arr):
    if arr.ndim == 1:
        return arr.reshape((-1, 1))
    return arr


def ensure_args_atleast_2d(f):
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        _args = []
        for arg in args:
            if isinstance(arg, list):
                _args.append([atleast_2d(_arr) for _arr in arg])
            elif isinstance(arg, np.ndarray)\
                    or isinstance(arg, torch.Tensor):
                _args.append(atleast_2d(arg))
            else:
                _args.append(arg)

        return f(self, *_args, **kwargs)
    return wrapper

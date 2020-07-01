import numpy as np
import torch

from functools import wraps


def transform(arr, trans=None):
    if trans is None:
        return arr
    else:
        _arr = trans.transform(arr)
        if isinstance(arr, np.ndarray):
            return _arr
        else:
            return to_float(_arr)


def inverse_transform(arr, trans=None):
    if trans is None:
        return arr
    else:
        _arr = trans.inverse_transform(arr)
        if isinstance(arr, np.ndarray):
            return _arr
        else:
            return to_float(_arr)


def atleast_2d(arr):
    if arr.ndim == 1:
        return arr.reshape((-1, 1))
    return arr


def to_float(arr):
    if isinstance(arr, np.ndarray):
        return torch.from_numpy(arr).float()
    elif isinstance(arr, torch.FloatTensor):
        return arr
    else:
        raise TypeError


def np_float(arr):
    if isinstance(arr, torch.Tensor):
        return arr.numpy()
    elif isinstance(arr, np.ndarray):
        return arr
    else:
        raise TypeError


def ensure_args_torch_floats(f):
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        _args = []
        for arg in args:
            if isinstance(arg, list):
                _args.append([to_float(_arr) for _arr in arg])
            elif isinstance(arg, np.ndarray):
                _args.append(to_float(arg))
            else:
                _args.append(arg)

        return f(self, *_args, **kwargs)
    return wrapper


def ensure_res_numpy_floats(f):
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        outputs = f(self, *args, **kwargs)

        _outputs = []
        for out in outputs:
            if isinstance(out, torch.Tensor):
                _outputs.append(np_float(out))
            elif isinstance(out, list):
                _outputs.append([np_float(x) for x in out])

        return _outputs
    return wrapper


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

import numpy as np
import torch

from functools import wraps


def transform(arr, trans=None):
    if trans is None:
        return arr
    else:
        if arr.ndim == 1:
            _size = trans.n_components
            _arr = trans.transform(arr.reshape(-1, _size))
            _arr = np.atleast_1d(np.squeeze(_arr))
        elif arr.ndim == 2:
            _arr = trans.transform(arr)
        elif arr.ndim == 3:
            _arr = np.stack([trans.transform(arr[i, ...])
                             for i in range(arr.shape[0])], axis=0)
        else:
            raise NotImplementedError

        if isinstance(arr, np.ndarray):
            return _arr
        else:
            if isinstance(arr, torch.FloatTensor):
                return to_float(_arr)
            else:
                return to_double(_arr)


def inverse_transform(arr, trans=None):
    if trans is None:
        return arr
    else:
        if arr.ndim == 1:
            _size = trans.n_components
            _arr = trans.inverse_transform(arr.reshape(-1, _size))
            _arr = np.atleast_1d(np.squeeze(_arr))
        elif arr.ndim == 2:
            _arr = trans.inverse_transform(arr)
        elif arr.ndim == 3:
            _arr = np.stack([trans.inverse_transform(arr[i, ...])
                             for i in range(arr.shape[0])], axis=0)
        else:
            raise NotImplementedError

        if isinstance(arr, np.ndarray):
            return _arr
        else:
            if isinstance(arr, torch.FloatTensor):
                return to_float(_arr)
            else:
                return to_double(_arr)


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
        pass


def to_double(arr):
    if isinstance(arr, np.ndarray):
        return torch.from_numpy(arr).double()
    elif isinstance(arr, torch.DoubleTensor):
        return arr
    else:
        pass


def np_float(arr):
    if isinstance(arr, torch.Tensor):
        return arr.numpy()
    elif isinstance(arr, np.ndarray):
        return arr
    else:
        pass


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


def ensure_args_torch_doubles(f):
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        _args = []
        for arg in args:
            if isinstance(arg, list):
                _args.append([to_double(_arr) for _arr in arg])
            elif isinstance(arg, np.ndarray):
                _args.append(to_double(arg))
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

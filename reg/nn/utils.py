import autograd.numpy as np
import torch

from itertools import islice
from functools import wraps

import random


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


def batches(batch_size, data_size):
    idx_all = random.sample(range(data_size), data_size)
    idx_iter = iter(idx_all)
    yield from iter(lambda: list(islice(idx_iter, batch_size)), [])


def relu(x):
    h = np.maximum(0, x)
    dh = np.where(x > 0, 1.0, 0.0)
    return h, dh


def logistic(x):
    h = 1.0 / (1.0 + np.exp(-x))
    dh = h * (1.0 - h)
    return h, dh


def linear(x):
    h, dh = x, 1.0
    return h, dh


def tanh(x):
    h = np.tanh(x)
    dh = 1.0 - h**2
    return h, dh


# mean squared error
def mse(target, output):
    l = 0.5 * np.mean(np.einsum('nk,nh->n', target - output, target - output), axis=0)
    dl = - (target - output)
    return l, dl


# cross entropy
def ce(target, output):
    out = np.clip(output, 1e-16, 1.0 - 1e-16)
    l = - np.mean(target * np.log(output) + (1.0 - target) * np.log(1.0 - out), axis=(0, 1))
    dl = (output - target) / ((1.0 - output) * output)
    return l, dl

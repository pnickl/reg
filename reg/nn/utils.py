import autograd.numpy as np
from itertools import islice
import random


def batches(batch_size, data_size):
    """
    :param batch_size: number of samples in each mini-batch
    :param data_size: total number of samples
    :return: generator of lists of indices
    """
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

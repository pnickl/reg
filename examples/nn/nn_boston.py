import autograd.numpy as np
import torch

from reg.nn.nn_ag import NNRegressor as agNetwork
from reg.nn.nn_torch import NNRegressor as torchNetwork
from reg.nn.nn_npy import NNRegressor as npNetwork


if __name__ == '__main__':

    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split

    # prepare data
    set = load_boston()

    x, y = set['data'], set['target']
    y = y[:, np.newaxis]

    xt, xv, yt, yv = train_test_split(x, y, test_size=0.2)

    nb_in = x.shape[-1]
    nb_out = y.shape[-1]

    # fit neural network with numpy
    nn_np = npNetwork([nb_in, 10, nb_out], nonlin='relu', output='linear', loss='mse')

    print('using numpy network:')
    nn_np.fit(yt, xt, nb_epochs=250, batch_size=64, lr=1e-2)

    print('numpy', 'train:', 'cost=', nn_np.cost(yt, xt)[0])
    print('numpy', 'test:', 'cost=', nn_np.cost(yv, xv)[0])

    # fit neural network with autograd
    nn_ag = agNetwork([nb_in, 10, nb_out], nonlin='relu', output='linear', loss='mse')

    print('using autograd network:')
    nn_ag.fit(yt, xt, nb_epochs=250, batch_size=64, lr=1e-2)

    print('autograd', 'train:', 'cost=', nn_ag.cost(yt, xt))
    print('autograd', 'test:', 'cost=', nn_ag.cost(yv, xv))

    # fit neural network with pytorch
    xt = torch.from_numpy(xt).float()
    yt = torch.from_numpy(yt).float()
    xv = torch.from_numpy(xv).float()
    yv = torch.from_numpy(yv).float()

    nn_torch = torchNetwork([nb_in, 10, nb_out], nonlin='relu')

    print('using pytorch network:')
    nn_torch.fit(yt, xt, nb_epochs=1000, batch_size=64, lr=1e-2)

    _yt = nn_torch.forward(xt)
    _yv = nn_torch.forward(xv)
    print('pytorch', 'train:', 'cost=', torch.mean(nn_torch.loss(yt, _yt)))
    print('pytorch', 'test:', 'cost=', torch.mean(nn_torch.loss(yv, _yv)))

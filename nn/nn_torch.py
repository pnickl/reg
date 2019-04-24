import autograd.numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.loss import MSELoss
from torch.optim import SGD, Adam

from reg.nn.util import batches


class Network(nn.Module):
    def __init__(self, sizes, nonlin='tanh'):

        super(Network, self).__init__()
        self.sizes = sizes

        nlist = dict(relu=F.relu, tanh=F.tanh,
            softmax=F.log_softmax, linear=F.linear)

        self.nonlin = nlist[nonlin]
        self.layers = [nn.Linear(x, y) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

        self.loss = MSELoss()

    def forward(self, x):
        _out = x
        for l in self.layers:
            _out = self.nonlin(l(_out))
        return _out

    def fit(self, y, x, nb_epochs, batch_size=32, lr=1e-3):
        self.opt = Adam([{'params': l.parameters()} for l in self.layers], lr=lr)

        for n in range(nb_epochs):
            for batch in batches(batch_size, y.shape[0]):
                self.opt.zero_grad()
                _out = self.forward(x[batch])
                loss = self.loss(_out, y[batch])
                loss.backward()
                self.opt.step()

            _y = nn.forward(x)
            print('iter=', n, 'cost=', torch.mean(nn.loss(y, _y)))


if __name__ == '__main__':

    # npr.seed(1337)

    from sklearn.datasets import load_breast_cancer, load_digits, load_boston

    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder

    # regression example
    set = load_boston()
    x, y = set['data'], set['target']
    y = y[:, np.newaxis]

    xt, xv, yt, yv = train_test_split(x, y, test_size=0.2)

    nb_in = x.shape[-1]
    nb_out = y.shape[-1]

    xt = torch.tensor(xt, dtype=torch.float)
    yt = torch.tensor(yt, dtype=torch.float)
    xv = torch.tensor(xv, dtype=torch.float)
    yv = torch.tensor(yv, dtype=torch.float)

    nn = Network([nb_in, 10, nb_out], nonlin='relu')
    nn.fit(yt, xt, nb_epochs=250, batch_size=64, lr=1e-2)

    _yt = nn.forward(xt)
    print('train:', 'cost=', torch.mean(nn.loss(yt, _yt)))

    _yv = nn.forward(xv)
    print('test:', 'cost=', torch.mean(nn.loss(yv, _yv)))

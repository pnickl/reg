import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.loss import MSELoss
from torch.optim import Adam

from reg.nn.utils import batches


class NNRegressor(nn.Module):
    def __init__(self, sizes, nonlin='tanh'):
        super(NNRegressor, self).__init__()

        self.sizes = sizes

        nlist = dict(relu=F.relu, tanh=F.tanh,
                     softmax=F.log_softmax, linear=F.linear)

        self.nonlin = nlist[nonlin]
        self.layers = [nn.Linear(x, y) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

        self.criterion = MSELoss()
        self.optim = None

    def forward(self, x):
        out = x
        for l in self.layers:
            out = self.nonlin(l(out))
        return out

    def fit(self, y, x, nb_epochs, batch_size=32, lr=1e-3):
        self.optim = Adam([{'params': l.parameters()} for l in self.layers], lr=lr)

        for n in range(nb_epochs):
            for batch in batches(batch_size, y.shape[0]):
                self.optim.zero_grad()
                _y = self.forward(x[batch])
                loss = self.criterion(_y, y[batch])
                loss.backward()
                self.optim.step()

            if n % 10 == 0:
                _y = self.forward(x)
                print('Epoch: {}/{}.............'.format(n, nb_epochs), end=' ')
                print("Loss: {:.4f}".format(torch.mean(self.criterion(y, _y))))

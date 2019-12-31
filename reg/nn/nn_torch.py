import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.loss import MSELoss
from torch.optim import Adam

import numpy as np

from reg.nn.utils import batches

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")


class NNRegressor(nn.Module):
    def __init__(self, sizes, nonlin='tanh'):
        super(NNRegressor, self).__init__()

        self.sizes = sizes

        nlist = dict(relu=torch.relu, tanh=torch.tanh,
                     softmax=torch.log_softmax, linear=F.linear)

        self.nonlin = nlist[nonlin]
        self.l1 = nn.Linear(self.sizes[0], self.sizes[1]).to(device)
        self.l2 = nn.Linear(self.sizes[1], self.sizes[2]).to(device)
        self.output = nn.Linear(self.sizes[2], self.sizes[3]).to(device)

        self.criterion = MSELoss().to(device)
        self.optim = None

    def forward(self, x):
        out = x
        out = self.nonlin(self.l1(out))
        out = self.nonlin(self.l2(out))
        return self.output(out)

    def fit(self, y, x, nb_epochs, batch_size=32, lr=1e-3):
        y = y.to(device)
        x = x.to(device)

        self.optim = Adam(self.parameters(), lr=lr)

        for n in range(nb_epochs):
            for batch in batches(batch_size, y.shape[0]):
                self.optim.zero_grad()
                _y = self.forward(x[batch])
                loss = self.criterion(_y, y[batch])
                loss.backward()
                self.optim.step()

            # if n % 10 == 0:
            #     _y = self.forward(x)
            #     print('Epoch: {}/{}.............'.format(n, nb_epochs), end=' ')
            #     print("Loss: {:.4f}".format(torch.mean(self.criterion(y, _y))))

    def forcast(self, x, horizon=1):
        x = x.to(device)

        with torch.no_grad():
            buffer_size = x.size(0)

            for t in range(buffer_size):
                _y = self(x[t, :].view(1, -1))

            yhat, _yhat = [_y], _y
            for _ in range(horizon):
                _yhat = self(_yhat.view(1, -1))
                yhat.append(_yhat)

            yhat = torch.stack(yhat, 0).view(horizon + 1, -1)
        return yhat


class DynamicNNRegressor(NNRegressor):
    def __init__(self, sizes, nonlin='tanh'):
        super(DynamicNNRegressor, self).__init__(sizes, nonlin)

    def forcast(self, x, u, horizon=1):
        x = x.to(device)
        u = u.to(device)

        with torch.no_grad():
            _yhat = x.view(1, -1)
            yhat = [x.view(1, -1)]
            for h in range(horizon):
                _u = u[h, :].view(1, -1)
                _in = torch.cat((_yhat, _u), 1)
                _yhat = self(_in)
                yhat.append(_yhat)

            yhat = torch.stack(yhat, 0).view(horizon + 1, -1)
        return yhat.cpu()

    def kstep_mse(self, y, x, u, horizon):
        from sklearn.metrics import mean_squared_error, explained_variance_score

        mse, evar = [], []
        for _x, _u, _y in zip(x, u, y):
            _target, _prediction = [], []
            for t in range(_x.shape[0] - horizon + 1):
                _yhat = self.forcast(_x[t, :], _u[t:t + horizon, :], horizon)

                # -1 because y is just x shifted by +1
                _target.append(_y.numpy()[t + horizon - 1, :])
                _prediction.append(_yhat.numpy()[-1, :])

            _target = np.vstack(_target)
            _prediction = np.vstack(_prediction)

            _mse = mean_squared_error(_target, _prediction)
            mse.append(_mse)

            _evar = explained_variance_score(_target, _prediction,
                                             multioutput='variance_weighted')
            evar.append(_evar)

        return np.mean(mse), np.mean(evar)

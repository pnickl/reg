import torch
import torch.nn as nn
from torch.optim import LBFGS

import numpy as np


class LSTMRegressor(nn.Module):
    def __init__(self, input_size, target_size, nb_neurons):
        super(LSTMRegressor, self).__init__()

        self.input_size = input_size
        self.target_size = target_size
        self.nb_neurons = nb_neurons
        self.nb_layers = 2

        self.l1 = nn.LSTMCell(self.input_size, self.nb_neurons[0])
        self.l2 = nn.LSTMCell(self.nb_neurons[0], self.nb_neurons[1])

        self.linear = nn.Linear(self.nb_neurons[1], self.target_size)

        self.criterion = nn.MSELoss()
        self.optim = None

    def init_hidden(self, batch_size):
        ht = torch.zeros(batch_size, self.nb_neurons[0], dtype=torch.double)
        ct = torch.zeros(batch_size, self.nb_neurons[0], dtype=torch.double)

        gt = torch.zeros(batch_size, self.nb_neurons[1], dtype=torch.double)
        bt = torch.zeros(batch_size, self.nb_neurons[1], dtype=torch.double)

        return ht, ct, gt, bt

    def forward(self, x):
        batch_size = x.size(0)
        ht, ct, gt, bt = self.init_hidden(batch_size)

        y = []
        for n in range(x.size(1)):
            ht, ct = self.l1(x[:, n, :], (ht, ct))
            gt, bt = self.l2(ht, (gt, bt))
            _y = self.linear(gt)
            y.append(_y)

        y = torch.stack(y, 1)
        return y

    def fit(self, y, x, nb_epochs, lr=0.5):
        self.double()

        self.optim = LBFGS(self.parameters(), lr=lr)
        for n in range(nb_epochs):

            def closure():
                self.optim.zero_grad()
                _y = self(x)
                loss = self.criterion(_y, y)
                loss.backward()

                print('Epoch: {}/{}.............'.format(n, nb_epochs), end=' ')
                print("Loss: {:.6f}".format(loss.item()))

                return loss

            self.optim.step(closure)

    def forcast(self, x, horizon=1):
        with torch.no_grad():
            buffer_size = x.size(0)
            ht, ct, gt, bt = self.init_hidden(1)

            for t in range(buffer_size):
                ht, ct = self.l1(x[t, :].view(1, -1), (ht, ct))
                gt, bt = self.l2(ht, (gt, bt))
                _y = self.linear(gt)

            yhat, _yhat = [_y], _y
            for _ in range(horizon):
                ht, ct = self.l1(_yhat, (ht, ct))
                gt, bt = self.l2(ht, (gt, bt))
                _yhat = self.linear(gt)
                yhat.append(_yhat)

            yhat = torch.stack(yhat, 0).view(horizon + 1, -1)
        return yhat


class DynamicLSTMRegressor(LSTMRegressor):
    def __init__(self, input_size, target_size, nb_neurons):
        super(DynamicLSTMRegressor, self).__init__(input_size, target_size, nb_neurons)

    def forcast(self, x, u, horizon=1):
        with torch.no_grad():
            buffer_size = x.size(0)
            ht, ct, gt, bt = self.init_hidden(1)

            for t in range(buffer_size):
                _u = u[t, :].view(1, -1)
                _x = x[t, :].view(1, -1)
                _in = torch.cat((_x, _u), 1)
                ht, ct = self.l1(_in, (ht, ct))
                gt, bt = self.l2(ht, (gt, bt))
                _y = self.linear(gt)

            yhat, _yhat = [_y], _y
            for h in range(horizon):
                _u = u[buffer_size + h, :].view(1, -1)
                _in = torch.cat((_yhat, _u), 1)
                ht, ct = self.l1(_in, (ht, ct))
                gt, bt = self.l2(ht, (gt, bt))
                _yhat = self.linear(gt)
                yhat.append(_yhat)

            yhat = torch.stack(yhat, 0).view(horizon + 1, -1)
        return yhat

    def kstep_mse(self, y, x, u, horizon):
        from sklearn.metrics import mean_squared_error, r2_score

        mse, norm_mse = [], []
        for _x, _u, _y in zip(x, u, y):
            _target, _prediction = [], []
            for t in range(_x.shape[0] - horizon):
                _yhat = self.forcast(_x[:t + 1, :], _u[:t + 1 + horizon, :], horizon)

                _target.append(_y.numpy()[t + horizon - 1, :])
                _prediction.append(_yhat.numpy()[-1, :])

            _target = np.vstack(_target)
            _prediction = np.vstack(_prediction)

            _mse = mean_squared_error(_target, _prediction)
            mse.append(_mse)

            _norm_mse = r2_score(_target, _prediction,
                                 multioutput='variance_weighted')
            norm_mse.append(_norm_mse)

        return np.mean(mse), np.mean(norm_mse)

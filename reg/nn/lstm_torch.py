import torch
import torch.nn as nn
from torch.optim import LBFGS, Adam

import numpy as np

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")


class LSTMRegressor(nn.Module):
    def __init__(self, input_size, target_size, nb_neurons):
        super(LSTMRegressor, self).__init__()

        self.input_size = input_size
        self.target_size = target_size
        self.nb_neurons = nb_neurons
        self.nb_layers = 2

        self.l1 = nn.LSTMCell(self.input_size, self.nb_neurons[0]).to(device)
        self.l2 = nn.LSTMCell(self.nb_neurons[0], self.nb_neurons[1]).to(device)

        self.linear = nn.Linear(self.nb_neurons[1], self.target_size).to(device)

        self.criterion = nn.MSELoss().to(device)
        self.optim = None

    def init_hidden(self, batch_size):
        ht = torch.zeros(batch_size, self.nb_neurons[0], dtype=torch.double).to(device)
        ct = torch.zeros(batch_size, self.nb_neurons[0], dtype=torch.double).to(device)

        gt = torch.zeros(batch_size, self.nb_neurons[1], dtype=torch.double).to(device)
        bt = torch.zeros(batch_size, self.nb_neurons[1], dtype=torch.double).to(device)

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
        y = y.to(device)
        x = x.to(device)

        self.double()

        self.optim = LBFGS(self.parameters(), lr=lr)
        # self.optim = Adam(self.parameters(), lr=lr)
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
        x = x.to(device)

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
        return yhat.cpu()


class DynamicLSTMRegressor(LSTMRegressor):
    def __init__(self, input_size, target_size, nb_neurons):
        super(DynamicLSTMRegressor, self).__init__(input_size, target_size, nb_neurons)

    def forcast(self, x, u, horizon=1):
        x = x.to(device)
        u = u.to(device)

        with torch.no_grad():
            buffer_size = x.size(0) - 1
            ht, ct, gt, bt = self.init_hidden(1)

            if buffer_size == 0:
                # no history
                _y = x[0, :].view(1, -1)
            else:
                # history
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
        return yhat.cpu()

    def kstep_mse(self, y, x, u, horizon):
        from sklearn.metrics import mean_squared_error, explained_variance_score

        mse, evar = [], []
        for _x, _u, _y in zip(x, u, y):
            _target, _prediction = [], []
            for t in range(_x.shape[0] - horizon + 1):
                _yhat = self.forcast(_x[:t + 1, :], _u[:t + horizon, :], horizon)

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

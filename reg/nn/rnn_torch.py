import torch
from torch import nn
from torch.optim import Adam

import numpy as np


class RNNRegressor(nn.Module):
    def __init__(self, input_size, target_size,
                 hidden_size, nb_layers,
                 nonlinearity='tanh'):
        super(RNNRegressor, self).__init__()

        self.input_size = input_size
        self.target_size = target_size
        self.hidden_size = hidden_size
        self.nb_layers = nb_layers

        self.nonlinearity = nonlinearity

        self.rnn = nn.RNN(input_size, hidden_size,
                          nb_layers, batch_first=True,
                          nonlinearity=nonlinearity)

        self.linear = nn.Linear(hidden_size, target_size)

        self.criterion = nn.MSELoss()
        self.optim = None

    def init_hidden(self, batch_size):
        return torch.zeros(self.nb_layers, batch_size, self.hidden_size)

    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)

        out, hidden = self.rnn(x, hidden)
        out = self.linear(out)

        return out, hidden

    def fit(self, y, x, nb_epochs, lr=1.e-3):
        self.optim = Adam(self.parameters(), lr=lr)

        for n in range(nb_epochs):
            self.optim.zero_grad()
            _y, hidden = self(x)
            loss = self.criterion(_y.view(-1, self.target_size),
                                  y.view(-1, self.target_size))
            loss.backward()
            self.optim.step()

            if n % 10 == 0:
                print('Epoch: {}/{}.............'.format(n, nb_epochs), end=' ')
                print("Loss: {:.6f}".format(loss.item()))

    def forcast(self, x, horizon=1):
        with torch.no_grad():
            buffer_size = x.size(0)
            _ht = self.init_hidden(1)

            for t in range(buffer_size):
                _y, _ht = self.rnn(x[t, :].view(1, 1, -1), _ht)
                _y = self.linear(_y)

            yhat, _yhat = [_y], _y
            for _ in range(horizon):
                _yhat, _ht = self.rnn(_yhat, _ht)
                _yhat = self.linear(_yhat)
                yhat.append(_yhat)

            yhat = torch.stack(yhat, 0).view(horizon + 1, -1)
        return yhat


class DynamicRNNRegressor(RNNRegressor):
    def __init__(self, input_size, target_size,
                 hidden_size, nb_layers,
                 nonlinearity='tanh'):
        super(DynamicRNNRegressor, self).__init__(input_size, target_size,
                                                  hidden_size, nb_layers,
                                                  nonlinearity)

    def forcast(self, x, u, horizon=1):
        with torch.no_grad():
            buffer_size = x.size(0) - 1
            _ht = self.init_hidden(1)

            if buffer_size == 0:
                # no history
                _y = x[0, :].view(1, -1)
            else:
                # history
                for t in range(buffer_size):
                    _u = u[t, :].view(1, 1, -1)
                    _x = x[t, :].view(1, 1, -1)
                    _in = torch.cat((_x, _u), 2)
                    _y, _ht = self.rnn(_in, _ht)
                    _y = self.linear(_y)

            yhat, _yhat = [_y], _y
            for h in range(horizon):
                _u = u[buffer_size + h, :].view(1, 1, -1)
                _in = torch.cat((_yhat, _u), 2)
                _yhat, _ht = self.rnn(_in, _ht)
                _yhat = self.linear(_yhat)
                yhat.append(_yhat)

            yhat = torch.stack(yhat, 0).view(horizon + 1, -1)
        return yhat

    def kstep_mse(self, y, x, u, horizon):
        from sklearn.metrics import mean_squared_error, explained_variance_score

        mse, norm_mse = [], []
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

            _norm_mse = explained_variance_score(_target, _prediction,
                                 multioutput='variance_weighted')
            norm_mse.append(_norm_mse)

        return np.mean(mse), np.mean(norm_mse)

import torch
from torch import nn
from torch.optim import Adam

import numpy as np


class RNNRegressor(nn.Module):
    def __init__(self, input_size, target_size,
                 hidden_size, nb_layers,
                 nonlinearity='tanh', device='cpu'):
        super(RNNRegressor, self).__init__()

        if device == 'gpu' and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        self.input_size = input_size
        self.target_size = target_size
        self.hidden_size = hidden_size
        self.nb_layers = nb_layers

        self.nonlinearity = nonlinearity

        self.rnn = nn.RNN(input_size, hidden_size,
                          nb_layers, batch_first=True,
                          nonlinearity=nonlinearity).to(self.device)

        self.linear = nn.Linear(hidden_size, target_size).to(self.device)

        self.criterion = nn.MSELoss().to(self.device)
        self.optim = None

    def init_hidden(self, batch_size):
        return torch.zeros(self.nb_layers, batch_size, self.hidden_size).to(self.device)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        hidden = self.init_hidden(batch_size)

        output, hidden = self.rnn(inputs, hidden)
        output = self.linear(output)

        return output, hidden

    def fit(self, target, input, nb_epochs, lr=1e-3, verbose=True):
        target = target.to(self.device)
        input = input.to(self.device)

        self.optim = Adam(self.parameters(), lr=lr)

        for n in range(nb_epochs):
            self.optim.zero_grad()
            _output, hidden = self(input)
            loss = self.criterion(_output.view(-1, self.target_size),
                                  target.view(-1, self.target_size))
            loss.backward()
            self.optim.step()

            if verbose:
                if n % 10 == 0:
                    print('Epoch: {}/{}.............'.format(n, nb_epochs), end=' ')
                    print("Loss: {:.6f}".format(loss.item()))

    def forcast(self, state, exogenous=None, horizon=1):
        assert exogenous is None
        state = state.to(self.device)

        with torch.no_grad():
            buffer_size = state.size(0)
            hidden = self.init_hidden(1)

            for t in range(buffer_size):
                _state, hidden = self.rnn(state[t, :].view(1, 1, -1), hidden)
                _state = self.linear(_state)

            forcast = [_state]
            for _ in range(horizon):
                _state, hidden = self.rnn(_state, hidden)
                _state = self.linear(_state)
                forcast.append(_state)

            forcast = torch.stack(forcast, 0).view(horizon + 1, -1)
        return forcast


class DynamicRNNRegressor(RNNRegressor):
    def __init__(self, input_size, target_size,
                 hidden_size, nb_layers,
                 nonlinearity='tanh', device='cpu'):
        super(DynamicRNNRegressor, self).__init__(input_size, target_size,
                                                  hidden_size, nb_layers,
                                                  nonlinearity, device)

    def forcast(self, state, exogenous=None, horizon=1):
        state = state.to(self.device)
        exogenous = exogenous.to(self.device)

        with torch.no_grad():
            buffer_size = state.size(0) - 1
            hidden = self.init_hidden(1)

            if buffer_size == 0:
                # no history
                _state = state[0, :].view(1, -1)
            else:
                # history
                for t in range(buffer_size):
                    _exo = exogenous[t, :].view(1, 1, -1)
                    _hist = state[t, :].view(1, 1, -1)
                    _input = torch.cat((_hist, _exo), 2)
                    _state, hidden = self.rnn(_input, hidden)
                    _state = self.linear(_state)

            forcast = [_state]
            for h in range(horizon):
                _exo = exogenous[buffer_size + h, :].view(1, 1, -1)
                _input = torch.cat((_state, _exo), 2)
                _state, hidden = self.rnn(_input, hidden)
                _state = self.linear(_state)
                forcast.append(_state)

            forcast = torch.stack(forcast, 0).view(horizon + 1, -1)
        return forcast.cpu()

    def kstep_mse(self, state, exogenous, horizon):
        from sklearn.metrics import mean_squared_error,\
            explained_variance_score, r2_score

        mse, smse, evar = [], [], []
        for _state, _exogenous in zip(state, exogenous):
            target, forcast = [], []

            nb_steps = _state.shape[0] - horizon + 1
            for t in range(nb_steps):
                _forcast = self.forcast(_state[t, :], _exogenous[t:t + horizon, :], horizon)

                target.append(_state.numpy()[t + horizon, :])
                forcast.append(_forcast.numpy()[-1, :])

            target = np.vstack(target)
            forcast = np.vstack(forcast)

            _mse = mean_squared_error(target, forcast)
            _smse = 1. - r2_score(target, forcast, multioutput='variance_weighted')
            _evar = explained_variance_score(target, forcast, multioutput='variance_weighted')

            mse.append(_mse)
            smse.append(_smse)
            evar.append(_evar)

        return np.mean(mse), np.mean(smse), np.mean(evar)

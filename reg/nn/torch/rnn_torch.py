import torch
from torch import nn
from torch.optim import Adam

import numpy as np

from sklearn.preprocessing import StandardScaler

from reg.nn.torch.utils import transform, inverse_transform
from reg.nn.torch.utils import ensure_args_torch_floats
from reg.nn.torch.utils import ensure_res_numpy_floats
from reg.nn.torch.utils import atleast_2d, atleast_3d


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

        self.input_trans = None
        self.target_trans = None

    @property
    def model(self):
        return self

    def init_hidden(self, batch_size):
        return torch.zeros(self.nb_layers, batch_size,
                           self.hidden_size).to(self.device)

    def forward(self, inputs, hidden=None):
        output, hidden = self.rnn(inputs, hidden)
        output = self.linear(output)
        return output, hidden

    def init_preprocess(self, target, input):
        self.target_trans = StandardScaler()
        self.input_trans = StandardScaler()

        self.target_trans.fit(target.reshape(-1, self.target_size))
        self.input_trans.fit(input.reshape(-1, self.input_size))

    @ensure_args_torch_floats
    def fit(self, target, input, nb_epochs, lr=1e-3,
            l2=1e-32, verbose=True, preprocess=True):

        if preprocess:
            self.init_preprocess(target, input)
            target = transform(target, self.target_trans)
            input = transform(input, self.input_trans)

        target = target.to(self.device)
        input = input.to(self.device)

        self.optim = Adam(self.parameters(), lr=lr, weight_decay=l2)

        for n in range(nb_epochs):
            self.optim.zero_grad()
            _output, hidden = self(atleast_3d(input, self.input_size))
            loss = self.criterion(atleast_3d(_output, self.target_size),
                                  atleast_3d(target, self.target_size))
            loss.backward()
            self.optim.step()

            if verbose:
                if n % 50 == 0:
                    output, _ = self.forward(atleast_3d(input, self.input_size))
                    print('Epoch: {}/{}.............'.format(n, nb_epochs), end=' ')
                    print("Loss: {:.6f}".format(self.criterion(atleast_3d(output, self.target_size),
                                                               atleast_3d(target, self.target_size))))

    @ensure_args_torch_floats
    @ensure_res_numpy_floats
    def predict(self, input, hidden):
        with torch.no_grad():
            input = transform(input, self.input_trans)
            output, hidden = self.forward(input.reshape(-1, 1, self.input_size), hidden)
            output = inverse_transform(output, self.target_trans)
        return output, hidden

    def forcast(self, state, exogenous=None, horizon=1):
        self.device = torch.device('cpu')
        self.model.to(self.device)

        assert exogenous is None

        _hidden = None

        if state.ndim < 3:
            state = atleast_3d(state, self.input_size)

        buffer_size = state.shape[1] - 1
        if buffer_size == 0:
            _state = state
        else:
            for t in range(buffer_size):
                _state, _hidden = self.predict(state[:, t, :], _hidden)

        forcast = [_state]
        for _ in range(horizon):
            _state, _hidden = self.predict(_state[:, -1, :], _hidden)
            forcast.append(_state)

        forcast = np.hstack(forcast)
        return forcast


class DynamicRNNRegressor(RNNRegressor):
    def __init__(self, input_size, target_size,
                 hidden_size, nb_layers,
                 nonlinearity='tanh', device='cpu'):
        super(DynamicRNNRegressor, self).__init__(input_size, target_size,
                                                  hidden_size, nb_layers,
                                                  nonlinearity, device)

    def forcast(self, state, exogenous=None, horizon=1):
        self.device = torch.device('cpu')
        self.model.to(self.device)

        _hidden = None

        if state.ndim < 3:
            state = atleast_3d(state, self.target_size)
        if exogenous.ndim < 3:
            exogenous = atleast_3d(exogenous, self.input_size - self.target_size)

        buffer_size = state.shape[1] - 1
        if buffer_size == 0:
            _state = state
        else:
            for t in range(buffer_size):
                _exo = exogenous[:, t, :]
                _hist = state[:, t, :]
                _input = np.hstack((_hist, _exo))
                _state, _hidden = self.predict(_input, _hidden)

        forcast = [_state]
        for h in range(horizon):
            _exo = exogenous[:, buffer_size + h, :]
            _hist = _state[:, -1, :]
            _input = np.hstack((_hist, _exo))
            _state, _hidden = self.predict(_input, _hidden)
            forcast.append(_state)

        forcast = np.hstack(forcast)
        return forcast

    def kstep_mse(self, state, exogenous, horizon):
        from sklearn.metrics import mean_squared_error,\
            explained_variance_score, r2_score

        mse, smse, evar = [], [], []
        for _state, _exogenous in zip(state, exogenous):
            target, forcast = [], []

            nb_steps = _state.shape[0] - horizon
            for t in range(nb_steps):
                _forcast = self.forcast(_state[:t + 1, :],
                                        _exogenous[:t + horizon, :],
                                        horizon)

                target.append(_state[t + horizon, :])
                forcast.append(_forcast[0, -1, :])

            target = np.vstack(target)
            forcast = np.vstack(forcast)

            _mse = mean_squared_error(target, forcast)
            _smse = 1. - r2_score(target, forcast, multioutput='variance_weighted')
            _evar = explained_variance_score(target, forcast, multioutput='variance_weighted')

            mse.append(_mse)
            smse.append(_smse)
            evar.append(_evar)

        return np.mean(mse), np.mean(smse), np.mean(evar)

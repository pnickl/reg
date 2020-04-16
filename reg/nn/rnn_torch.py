import torch
from torch import nn
from torch.optim import Adam

import numpy as np

from sklearn.decomposition import PCA

from reg.nn.utils import transform, inverse_transform
from reg.nn.utils import ensure_args_torch_floats
from reg.nn.utils import ensure_res_numpy_floats


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

    def init_hidden(self, batch_size):
        return torch.zeros(self.nb_layers, batch_size, self.hidden_size).to(self.device)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        hidden = self.init_hidden(batch_size)

        output, hidden = self.rnn(inputs, hidden)
        output = self.linear(output)

        return output, hidden

    def init_preprocess(self, target, input):
        target_size = target.shape[-1]
        input_size = input.shape[-1]

        self.target_trans = PCA(n_components=target_size, whiten=True)
        self.input_trans = PCA(n_components=input_size, whiten=True)

        self.target_trans.fit(target.reshape(-1, target_size))
        self.input_trans.fit(input.reshape(-1, input_size))

    @ensure_args_torch_floats
    def fit(self, target, input, nb_epochs, lr=1e-3,
            verbose=True, preprocess=True):

        if preprocess:
            self.init_preprocess(target, input)
            target = transform(target, self.target_trans)
            input = transform(input, self.input_trans)

        target = target.to(self.device)
        input = input.to(self.device)

        self.optim = Adam(self.parameters(), lr=lr)

        for n in range(nb_epochs):
            self.optim.zero_grad()
            _output, hidden = self(input)
            loss = self.criterion(_output.reshape(-1, self.target_size),
                                  target.reshape(-1, self.target_size))
            loss.backward()
            self.optim.step()

            if verbose:
                if n % 50 == 0:
                    output, _ = self.forward(input)
                    print('Epoch: {}/{}.............'.format(n, nb_epochs), end=' ')
                    print("Loss: {:.6f}".format(self.criterion(output.reshape(-1, self.target_size),
                                                               target.reshape(-1, self.target_size))))

    @ensure_args_torch_floats
    @ensure_res_numpy_floats
    def predict(self, input, hidden):
        input = transform(input, self.input_trans)

        if hidden is None:
            hidden = self.init_hidden(1)

        input = input.to(self.device)
        for _h in hidden:
            _h.to(self.device)

        _output, hidden = self.rnn(input.view(-1, 1, self.input_size), hidden)
        output = self.linear(_output)

        output = inverse_transform(output, self.target_trans).cpu()
        output = output.reshape((self.target_size, ))

        return output, hidden

    def forcast(self, state, exogenous=None, horizon=1):
        assert exogenous is None

        with torch.no_grad():
            _hidden = None

            if state.ndim == 1:
                state = np.atleast_2d(state)

            buffer_size = state.shape[0] - 1
            if buffer_size == 0:
                _state = state[0, :]
            else:
                for t in range(buffer_size):
                    _state, _hidden = self.predict(state[t, :], _hidden)

                forcast = [_state]
                for _ in range(horizon):
                    _state, _hidden = self.predict(_state, _hidden)
                    forcast.append(_state)

                forcast = np.vstack(forcast)
            return forcast


class DynamicRNNRegressor(RNNRegressor):
    def __init__(self, input_size, target_size,
                 hidden_size, nb_layers,
                 nonlinearity='tanh', device='cpu'):
        super(DynamicRNNRegressor, self).__init__(input_size, target_size,
                                                  hidden_size, nb_layers,
                                                  nonlinearity, device)

    def forcast(self, state, exogenous=None, horizon=1):

        with torch.no_grad():
            _hidden = None

            if state.ndim == 1:
                state = np.atleast_2d(state)

            buffer_size = state.shape[0] - 1
            if buffer_size == 0:
                _state = state[0, :]
            else:
                for t in range(buffer_size):
                    _exo = exogenous[t, :]
                    _hist = state[t, :]
                    _input = np.hstack((_hist, _exo))
                    _state, _hidden = self.predict(_input, _hidden)

            forcast = [_state]
            for h in range(horizon):
                _exo = exogenous[buffer_size + h, :]
                _input = np.hstack((_state, _exo))
                _state, _hidden = self.predict(_input, _hidden)
                forcast.append(_state)

            forcast = np.vstack(forcast)
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
                forcast.append(_forcast[-1, :])

            target = np.vstack(target)
            forcast = np.vstack(forcast)

            _mse = mean_squared_error(target, forcast)
            _smse = 1. - r2_score(target, forcast, multioutput='variance_weighted')
            _evar = explained_variance_score(target, forcast, multioutput='variance_weighted')

            mse.append(_mse)
            smse.append(_smse)
            evar.append(_evar)

        return np.mean(mse), np.mean(smse), np.mean(evar)

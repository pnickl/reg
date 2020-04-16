import torch
import torch.nn as nn
from torch.optim import LBFGS, Adam

import numpy as np

from sklearn.decomposition import PCA

from reg.nn.utils import transform, inverse_transform
from reg.nn.utils import ensure_args_torch_doubles
from reg.nn.utils import ensure_res_numpy_floats


class LSTMRegressor(nn.Module):
    def __init__(self, input_size, target_size,
                 nb_neurons, device='cpu'):
        super(LSTMRegressor, self).__init__()
        if device == 'gpu' and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        self.input_size = input_size
        self.target_size = target_size
        self.nb_neurons = nb_neurons
        self.nb_layers = 2

        self.l1 = nn.LSTMCell(self.input_size, self.nb_neurons[0]).to(self.device)
        self.l2 = nn.LSTMCell(self.nb_neurons[0], self.nb_neurons[1]).to(self.device)

        self.linear = nn.Linear(self.nb_neurons[1], self.target_size).to(self.device)

        self.criterion = nn.MSELoss().to(self.device)
        self.optim = None

        self.input_trans = None
        self.target_trans = None

    @property
    def model(self):
        return self

    def init_hidden(self, batch_size):
        ht = torch.zeros(batch_size, self.nb_neurons[0], dtype=torch.double).to(self.device)
        ct = torch.zeros(batch_size, self.nb_neurons[0], dtype=torch.double).to(self.device)

        gt = torch.zeros(batch_size, self.nb_neurons[1], dtype=torch.double).to(self.device)
        bt = torch.zeros(batch_size, self.nb_neurons[1], dtype=torch.double).to(self.device)

        return list([ht, ct, gt, bt])

    def forward(self, inputs):
        batch_size = inputs.size(0)
        hidden = self.init_hidden(batch_size)

        output = []
        for n in range(inputs.size(1)):
            hidden[0:2] = self.l1(inputs[:, n, :], hidden[0:2])
            hidden[2:] = self.l2(hidden[0], hidden[2:])
            _output = self.linear(hidden[2])
            output.append(_output)

        output = torch.stack(output, 1)
        return output

    def init_preprocess(self, target, input):
        target_size = target.shape[-1]
        input_size = input.shape[-1]

        self.target_trans = PCA(n_components=target_size, whiten=True)
        self.input_trans = PCA(n_components=input_size, whiten=True)

        self.target_trans.fit(target.reshape(-1, target_size))
        self.input_trans.fit(input.reshape(-1, input_size))

    @ensure_args_torch_doubles
    def fit(self, target, input, nb_epochs, lr=0.5,
            verbose=True, preprocess=True):

        if preprocess:
            self.init_preprocess(target, input)
            target = transform(target, self.target_trans)
            input = transform(input, self.input_trans)

        target = target.to(self.device)
        input = input.to(self.device)

        self.model.double()

        self.optim = LBFGS(self.parameters(), lr=lr)
        # self.optim = Adam(self.parameters(), lr=lr)

        for n in range(nb_epochs):

            def closure():
                self.optim.zero_grad()
                _output = self.model(input)
                loss = self.criterion(_output, target)
                loss.backward()

                return loss

            self.optim.step(closure)

            if verbose:
                if n % 1 == 0:
                    output = self.forward(input)
                    print('Epoch: {}/{}.............'.format(n, nb_epochs), end=' ')
                    print("Loss: {:.6f}".format(self.criterion(output, target)))

    @ensure_args_torch_doubles
    @ensure_res_numpy_floats
    def predict(self, input, hidden):
        input = transform(input, self.input_trans)

        if hidden is None:
            hidden = self.init_hidden(1)

        input = input.to(self.device)
        for _h in hidden:
            _h.to(self.device)

        hidden[0:2] = self.l1(input.view(1, -1), hidden[0:2])
        hidden[2:] = self.l2(hidden[0], hidden[2:])
        output = self.linear(hidden[2])

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


class DynamicLSTMRegressor(LSTMRegressor):
    def __init__(self, input_size, target_size,
                 nb_neurons, device='cpu'):
        super(DynamicLSTMRegressor, self).__init__(input_size, target_size,
                                                   nb_neurons, device)

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

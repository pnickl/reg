import torch
import torch.nn as nn
from torch.optim import LBFGS, Adam

import numpy as np


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

    @property
    def model(self):
        return self

    def init_hidden(self, batch_size):
        ht = torch.zeros(batch_size, self.nb_neurons[0], dtype=torch.double).to(self.device)
        ct = torch.zeros(batch_size, self.nb_neurons[0], dtype=torch.double).to(self.device)

        gt = torch.zeros(batch_size, self.nb_neurons[1], dtype=torch.double).to(self.device)
        bt = torch.zeros(batch_size, self.nb_neurons[1], dtype=torch.double).to(self.device)

        return ht, ct, gt, bt

    def forward(self, inputs):
        batch_size = inputs.size(0)
        ht, ct, gt, bt = self.init_hidden(batch_size)

        output = []
        for n in range(inputs.size(1)):
            ht, ct = self.l1(inputs[:, n, :], (ht, ct))
            gt, bt = self.l2(ht, (gt, bt))
            _output = self.linear(gt)
            output.append(_output)

        output = torch.stack(output, 1)
        return output

    def fit(self, target, input, nb_epochs, lr=0.5, verbose=True):
        target = target.to(self.device)
        input = input.to(self.device)

        self.model.double()

        self.optim = LBFGS(self.parameters(), lr=lr)
        # self.optim = Adam(self.parameters(), lr=lr)
        for n in range(nb_epochs):

            def closure():
                self.optim.zero_grad()
                _output = self(input)
                loss = self.criterion(_output, target)
                loss.backward()

                if verbose:
                    print('Epoch: {}/{}.............'.format(n, nb_epochs), end=' ')
                    print("Loss: {:.6f}".format(loss.item()))

                return loss

            self.optim.step(closure)

    def forcast(self, state, exogenous=None, horizon=1):
        assert exogenous is None
        state = state.to(self.device)

        with torch.no_grad():
            buffer_size = state.size(0)
            ht, ct, gt, bt = self.init_hidden(1)

            for t in range(buffer_size):
                ht, ct = self.l1(state[t, :].view(1, -1), (ht, ct))
                gt, bt = self.l2(ht, (gt, bt))
                _state = self.linear(gt)

            forcast = [_state]
            for _ in range(horizon):
                ht, ct = self.l1(_state, (ht, ct))
                gt, bt = self.l2(ht, (gt, bt))
                _state = self.linear(gt)
                forcast.append(_state)

            forcast = torch.stack(forcast, 0).view(horizon + 1, -1)
        return forcast.cpu()


class DynamicLSTMRegressor(LSTMRegressor):
    def __init__(self, input_size, target_size,
                 nb_neurons, device='cpu'):
        super(DynamicLSTMRegressor, self).__init__(input_size, target_size,
                                                   nb_neurons, device)

    def forcast(self, state, exogenous=None, horizon=1):
        state = state.to(self.device)
        exogenous = exogenous.to(self.device)

        with torch.no_grad():
            buffer_size = state.size(0) - 1
            ht, ct, gt, bt = self.init_hidden(1)

            if buffer_size == 0:
                # no history
                _state = state[0, :].view(1, -1)
            else:
                # history
                for t in range(buffer_size):
                    _exo = exogenous[t, :].view(1, -1)
                    _hist = state[t, :].view(1, -1)
                    _input = torch.cat((_hist, _exo), 1)
                    ht, ct = self.l1(_input, (ht, ct))
                    gt, bt = self.l2(ht, (gt, bt))
                    _state = self.linear(gt)

            forcast = [_state]
            for h in range(horizon):
                _exo = exogenous[buffer_size + h, :].view(1, -1)
                _input = torch.cat((_state, _exo), 1)
                ht, ct = self.l1(_input, (ht, ct))
                gt, bt = self.l2(ht, (gt, bt))
                _state = self.linear(gt)
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

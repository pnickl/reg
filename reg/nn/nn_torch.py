import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.loss import MSELoss
from torch.optim import Adam

import numpy as np

from reg.nn.utils import batches


class NNRegressor(nn.Module):
    def __init__(self, sizes, nonlin='tanh', device='cpu'):
        super(NNRegressor, self).__init__()

        if device == 'gpu' and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        self.sizes = sizes

        nlist = dict(relu=torch.relu, tanh=torch.tanh,
                     softmax=torch.log_softmax, linear=F.linear)

        self.nonlin = nlist[nonlin]
        self.l1 = nn.Linear(self.sizes[0], self.sizes[1]).to(self.device)
        self.l2 = nn.Linear(self.sizes[1], self.sizes[2]).to(self.device)
        self.output = nn.Linear(self.sizes[2], self.sizes[3]).to(self.device)

        self.criterion = MSELoss().to(self.device)
        self.optim = None

    def forward(self, input):
        output = self.nonlin(self.l1(input))
        output = self.nonlin(self.l2(output))
        return self.output(output)

    def fit(self, target, input, nb_epochs, batch_size=32, lr=1e-3, l2=1e-16, verbose=True):
        target = target.to(self.device)
        input = input.to(self.device)

        self.optim = Adam(self.parameters(), lr=lr, weight_decay=l2)

        for n in range(nb_epochs):
            for batch in batches(batch_size, target.shape[0]):
                self.optim.zero_grad()
                _output = self.forward(input[batch])
                loss = self.criterion(_output, target[batch])
                loss.backward()
                self.optim.step()

            if n % 10 == 0:
                _output = self.forward(input)
                if verbose:
                    print('Epoch: {}/{}.............'.format(n, nb_epochs), end=' ')
                    print("Loss: {:.4f}".format(torch.mean(self.criterion(_output, target))))


class DynamicNNRegressor(NNRegressor):
    def __init__(self, sizes, nonlin='tanh',
                 incremental=True, device='cpu'):
        super(DynamicNNRegressor, self).__init__(sizes, nonlin, device)

        self.incremental = incremental

    def forcast(self, state, exogenous=None, horizon=1):
        state = state.to(self.device)
        exogenous = exogenous.to(self.device)

        with torch.no_grad():
            _state = state.view(1, -1)
            forcast = [_state]
            for h in range(horizon):
                _exo = exogenous[h, :].view(1, -1)
                _input = torch.cat((_state, _exo), 1)
                if self.incremental:
                    _state = _state + self(_input)
                else:
                    _state = self(_input)
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

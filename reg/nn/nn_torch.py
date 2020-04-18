import torch
import torch.nn as nn
import torch.nn.functional as func

from torch.nn.modules.loss import MSELoss
from torch.optim import Adam, SGD

import numpy as np

from sklearn.preprocessing import StandardScaler

from reg.nn.utils import batches, atleast_2d
from reg.gp.utils import transform, inverse_transform
from reg.gp.utils import ensure_args_torch_floats
from reg.gp.utils import ensure_res_numpy_floats


class NNRegressor(nn.Module):
    def __init__(self, sizes, nonlin='tanh', device='cpu'):
        super(NNRegressor, self).__init__()

        if device == 'gpu' and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        self.sizes = sizes

        nlist = dict(relu=torch.relu, tanh=torch.tanh, splus=nn.Softplus,
                     softmax=torch.log_softmax, linear=func.linear)

        self.nonlin = nlist[nonlin]
        self.l1 = nn.Linear(self.sizes[0], self.sizes[1]).to(self.device)
        self.l2 = nn.Linear(self.sizes[1], self.sizes[2]).to(self.device)
        self.output = nn.Linear(self.sizes[2], self.sizes[3]).to(self.device)

        self.criterion = MSELoss().to(self.device)
        self.optim = None

        self.target_size = self.sizes[-1]
        self.input_size = self.sizes[0]

        self.input_trans = None
        self.target_trans = None

    @ensure_args_torch_floats
    def forward(self, inputs):
        output = self.nonlin(self.l1(inputs))
        output = self.nonlin(self.l2(output))
        return torch.squeeze(self.output(output))

    @ensure_args_torch_floats
    @ensure_res_numpy_floats
    def predict(self, input):

        with torch.no_grad():
            input = transform(input, self.input_trans).to(self.device)
            input = atleast_2d(input, self.input_size)

            output = self.forward(input)
            output = inverse_transform(output.cpu(), self.target_trans)

        return output

    def init_preprocess(self, target, input):
        self.target_trans = StandardScaler()
        self.input_trans = StandardScaler()

        self.target_trans.fit(target.reshape(-1, self.target_size))
        self.input_trans.fit(input.reshape(-1, self.input_size))

    @ensure_args_torch_floats
    def fit(self, target, input, nb_epochs=1000, batch_size=32,
            lr=1e-3, l2=1e-32, verbose=True, preprocess=True):

        if preprocess:
            self.init_preprocess(target, input)
            target = transform(target, self.target_trans)
            input = transform(input, self.input_trans)

        target = target.to(self.device)
        input = input.to(self.device)

        self.optim = Adam(self.parameters(), lr=lr, weight_decay=l2)

        for n in range(nb_epochs):
            for batch in batches(batch_size, target.shape[0]):
                self.optim.zero_grad()
                _output = self.forward(atleast_2d(input[batch], self.input_size))
                loss = self.criterion(atleast_2d(_output, self.target_size),
                                      atleast_2d(target[batch], self.target_size))
                loss.backward()
                self.optim.step()

            if verbose:
                if n % 50 == 0:
                    output = self.forward(atleast_2d(input, self.input_size))
                    print('Epoch: {}/{}.............'.format(n, nb_epochs), end=' ')
                    print("Loss: {:.6f}".format(self.criterion(atleast_2d(output, self.target_size),
                                                               atleast_2d(target, self.target_size))))


class DynamicNNRegressor(NNRegressor):
    def __init__(self, sizes, nonlin='tanh',
                 incremental=True, device='cpu'):
        super(DynamicNNRegressor, self).__init__(sizes, nonlin, device)

        self.incremental = incremental

    @ensure_args_torch_floats
    @ensure_res_numpy_floats
    def predict(self, input):

        with torch.no_grad():
            _input = transform(input, self.input_trans).to(self.device)
            _input = atleast_2d(_input, self.input_size)

            output = self.forward(_input)
            output = inverse_transform(output.cpu(), self.target_trans)

        if self.incremental:
            return input[..., :self.target_size] + output
        else:
            return output

    def forcast(self, state, exogenous, horizon=1):
        _state = state
        forcast = [_state]
        for h in range(horizon):
            _exo = exogenous[h, :]
            _input = np.hstack((_state, _exo))
            _state = self.predict(_input)
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
                _forcast = self.forcast(_state[t, :],
                                        _exogenous[t:t + horizon, :],
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

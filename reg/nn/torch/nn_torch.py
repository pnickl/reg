import torch
import torch.nn as nn
import torch.nn.functional as func

from torch.utils.data import BatchSampler, SubsetRandomSampler

from torch.nn.modules.loss import MSELoss
from torch.optim import Adam

import numpy as np

from sklearn.decomposition import PCA

from reg.nn.torch.utils import transform, inverse_transform
from reg.nn.torch.utils import ensure_args_torch_floats
from reg.nn.torch.utils import ensure_res_numpy_floats
from reg.nn.torch.utils import ensure_args_atleast_2d


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

    @property
    def model(self):
        return self

    def forward(self, inputs):
        output = self.nonlin(self.l1(inputs))
        output = self.nonlin(self.l2(output))
        return self.output(output)

    @ensure_args_torch_floats
    @ensure_res_numpy_floats
    def predict(self, input):
        self.device = torch.device('cpu')
        self.model.to(self.device)

        input = input.reshape((-1, self.input_size))
        input = transform(input, self.input_trans)

        with torch.no_grad():
            output = self.forward(input).cpu()

        output = inverse_transform(output, self.target_trans)
        return torch.squeeze(output)

    def init_preprocess(self, target, input):
        self.target_trans = PCA(n_components=self.target_size, whiten=True)
        self.input_trans = PCA(n_components=self.input_size, whiten=True)

        self.target_trans.fit(target)
        self.input_trans.fit(input)

    @ensure_args_torch_floats
    @ensure_args_atleast_2d
    def fit(self, target, input, nb_epochs=1000, batch_size=32,
            lr=1e-3, l2=1e-32, verbose=True, preprocess=True):

        if preprocess:
            self.init_preprocess(target, input)
            target = transform(target, self.target_trans)
            input = transform(input, self.input_trans)

        target = target.to(self.device)
        input = input.to(self.device)

        set_size = input.shape[0]
        batch_size = set_size if batch_size is None else batch_size

        self.optim = Adam(self.parameters(), lr=lr, weight_decay=l2)

        for n in range(nb_epochs):
            batches = list(BatchSampler(SubsetRandomSampler(range(set_size)), batch_size, True))

            for batch in batches:
                self.optim.zero_grad()
                _output = self.forward(input[batch])
                loss = self.criterion(_output, target[batch])
                loss.backward()
                self.optim.step()

            if verbose:
                if n % 50 == 0:
                    output = self.forward(input)
                    print('Epoch: {}/{}.............'.format(n, nb_epochs), end=' ')
                    print("Loss: {:.6f}".format(self.criterion(output, target)))


class DynamicNNRegressor(NNRegressor):
    def __init__(self, sizes, nonlin='tanh',
                 incremental=True, device='cpu'):
        super(DynamicNNRegressor, self).__init__(sizes, nonlin, device)

        self.incremental = incremental

    @ensure_args_torch_floats
    @ensure_res_numpy_floats
    def predict(self, input):
        input = input.reshape((-1, self.input_size))
        input = transform(input, self.input_trans)

        with torch.no_grad():
                output = self.forward(input)

        output = inverse_transform(output, self.target_trans)

        if self.incremental:
            return input[..., :self.target_size] + output
        else:
            return output

    def forcast(self, state, exogenous, horizon=1):
        self.device = torch.device('cpu')
        self.model.to(self.device)

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

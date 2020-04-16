import numpy as np
import torch
from torch.optim import Adam

import gpytorch

from gpytorch.means import MultitaskMean, ZeroMean
from gpytorch.kernels import MultitaskKernel, RBFKernel
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import MultitaskGaussianLikelihood

from sklearn.decomposition import PCA

from reg.gp.utils import transform, inverse_transform
from reg.gp.utils import ensure_args_torch_floats
from reg.gp.utils import ensure_res_numpy_floats


class MultiGPRegressor(gpytorch.models.ExactGP):

    def __init__(self, target_size, device='cpu'):
        if device == 'gpu' and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        self.target_size = target_size

        _likelihood = MultitaskGaussianLikelihood(num_tasks=self.target_size)
        super(MultiGPRegressor, self).__init__(train_inputs=None,
                                               train_targets=None,
                                               likelihood=_likelihood)

        self.mean_module = MultitaskMean(ZeroMean(), num_tasks=self.target_size)
        self.covar_module = MultitaskKernel(RBFKernel(), num_tasks=self.target_size, rank=1)

        self.input_trans = None
        self.target_trans = None

    @property
    def model(self):
        return self

    def forward(self, input):
        mean = self.mean_module(input)
        covar = self.covar_module(input)
        output = MultitaskMultivariateNormal(mean, covar)
        return output

    @ensure_args_torch_floats
    @ensure_res_numpy_floats
    def predict(self, input):
        input = input.to(self.device)

        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            input = transform(input, self.input_trans)
            output = self.likelihood(self.model(torch.unsqueeze(input, 0))).mean
            output = inverse_transform(output, self.target_trans)

            output = output.reshape((self.target_size, ))
        return output

    def init_preprocess(self, target, input):
        target_size = target.shape[-1]
        input_size = input.shape[-1]

        self.target_trans = PCA(n_components=target_size, whiten=True)
        self.input_trans = PCA(n_components=input_size, whiten=True)

        self.target_trans.fit(target)
        self.input_trans.fit(input)

    @ensure_args_torch_floats
    def fit(self, target, input, nb_iter=100, lr=1e-1,
            verbose=True, preprocess=True):

        if preprocess:
            self.init_preprocess(target, input)
            target = transform(target, self.target_trans)
            input = transform(input, self.input_trans)

        target = target.to(self.device)
        input = input.to(self.device)

        self.model.set_train_data(input, target, strict=False)
        self.model.train().to(self.device)
        self.likelihood.train().to(self.device)

        optimizer = Adam([{'params': self.parameters()}], lr=lr)
        mll = ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(nb_iter):
            optimizer.zero_grad()
            _output = self.model(input)
            loss = - mll(_output, target)
            loss.backward()
            if verbose:
                print('Iter %d/%d - Loss: %.3f' % (i + 1, nb_iter, loss.item()))
            optimizer.step()


class DynamicMultiGPRegressor(MultiGPRegressor):
    def __init__(self, state_size, incremental=True, device='cpu'):
        super(DynamicMultiGPRegressor, self).__init__(state_size, device)

        self.incremental = incremental

    @ensure_args_torch_floats
    @ensure_res_numpy_floats
    def predict(self, input):

        input = input.to(self.device)

        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            _input = transform(input, self.input_trans)
            output = self.likelihood(self.model(torch.unsqueeze(_input, 0))).mean
            output = inverse_transform(output, self.target_trans)

            output = output.reshape((self.target_size, ))

        if self.incremental:
            return (input[:self.target_size] + output).cpu()
        else:
            return output.cpu()

    def forcast(self, state, exogenous, horizon=1):
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():
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

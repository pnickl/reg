import numpy as np
import torch
from torch.optim import Adam

import gpytorch

from gpytorch.means import MultitaskMean, ZeroMean
from gpytorch.kernels import MultitaskKernel, RBFKernel
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import MultitaskGaussianLikelihood


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

    @property
    def model(self):
        return self

    def forward(self, input):
        mean = self.mean_module(input)
        covar = self.covar_module(input)
        return MultitaskMultivariateNormal(mean, covar)

    def fit(self, target, input, nb_iter=100, lr=0.1, verbose=True):
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

    def forcast(self, state, exogenous, horizon=1):
        state = state.to(self.device)
        exogenous = exogenous.to(self.device)

        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            _state = state.view(1, -1)
            forcast = [_state]
            for h in range(horizon):
                _exo = exogenous[h, :].view(1, -1)
                _input = torch.cat((_state, _exo), 1)
                if self.incremental:
                    _state = _state + self.likelihood(self(_input)).mean
                else:
                    _state = self.likelihood(self(_input)).mean
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

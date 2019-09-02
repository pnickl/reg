import numpy as np
import torch
from torch.optim import Adam

import gpytorch

from gpytorch.means import MultitaskMean, ConstantMean, ZeroMean
from gpytorch.kernels import MultitaskKernel, RBFKernel
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import MultitaskGaussianLikelihood


class MultiGPRegressor(gpytorch.models.ExactGP):

    def __init__(self, input, target):
        self.input = input
        self.target = target

        self.target_size = target.size(1)

        _likelihood = MultitaskGaussianLikelihood(num_tasks=self.target_size)
        super(MultiGPRegressor, self).__init__(input, target, _likelihood)

        self.mean_module = MultitaskMean(ZeroMean(), num_tasks=self.target_size)
        self.covar_module = MultitaskKernel(RBFKernel(), num_tasks=self.target_size, rank=1)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal(mean_x, covar_x)

    def fit(self, nb_iter=100):
        self.train()
        self.likelihood.train()

        optimizer = Adam([{'params': self.parameters()}], lr=0.1)
        mll = ExactMarginalLogLikelihood(self.likelihood, self)

        for i in range(nb_iter):
            optimizer.zero_grad()
            _output = self(self.input)
            loss = - mll(_output, self.target)
            loss.backward()
            # print('Iter %d/%d - Loss: %.3f' % (i + 1, nb_iter, loss.item()))
            optimizer.step()


class DynamicMultiGPRegressor(MultiGPRegressor):
    def __init__(self, input, target):
        super(DynamicMultiGPRegressor, self).__init__(input, target)

    def forcast(self, x, u, horizon=0):
        self.eval()
        self.likelihood.eval()

        with torch.no_grad():
            _yhat = x.view(1, -1)
            yhat = [x.view(1, -1)]
            for h in range(horizon):
                _u = u[h, :].view(1, -1)
                _in = torch.cat((_yhat, _u), 1)
                _yhat = self.likelihood(self(_in)).mean
                yhat.append(_yhat)

            yhat = torch.stack(yhat, 0).view(horizon + 1, -1)
        return yhat

    def kstep_mse(self, y, x, u, horizon):
        from sklearn.metrics import mean_squared_error, explained_variance_score

        mse, norm_mse = [], []
        for _x, _u, _y in zip(x, u, y):
            _target, _prediction = [], []
            for t in range(_x.shape[0] - horizon + 1):
                _yhat = self.forcast(_x[t, :], _u[t:t + horizon, :], horizon)

                # -1 because y is just x shifted by +1
                _target.append(_y.numpy()[t + horizon - 1, :])
                _prediction.append(_yhat.numpy()[-1, :])

            _target = np.vstack(_target)
            _prediction = np.vstack(_prediction)

            _mse = mean_squared_error(_target, _prediction)
            mse.append(_mse)

            _norm_mse = explained_variance_score(_target, _prediction,
                                 multioutput='variance_weighted')
            norm_mse.append(_norm_mse)

        return np.mean(mse), np.mean(norm_mse)

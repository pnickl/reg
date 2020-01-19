import numpy as np
import torch
from torch.optim import Adam

import gpytorch

from gpytorch.means import MultitaskMean, ConstantMean, ZeroMean
from gpytorch.kernels import MultitaskKernel, RBFKernel
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import MultitaskGaussianLikelihood

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu" if torch.cuda.is_available() else "cpu")


class MultiGPRegressor(gpytorch.models.ExactGP):

    def __init__(self, input, target):
        self.input = input.to(device)
        self.target = target.to(device)

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
        self.train().to(device)
        self.likelihood.train().to(device)

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
    def __init__(self, input, target, incremental=True):
        super(DynamicMultiGPRegressor, self).__init__(input, target)

        self.incremental = incremental

    def forcast(self, x, u, horizon=0):
        x = x.to(device)
        u = u.to(device)

        self.eval()
        self.likelihood.eval()

        with torch.no_grad():
            _xn = x.view(1, -1)
            xn = [x.view(1, -1)]
            for h in range(horizon):
                _u = u[h, :].view(1, -1)
                _in = torch.cat((_xn, _u), 1)
                if self.incremental:
                    _xn = _xn + self.likelihood(self(_in)).mean
                else:
                    _xn = self.likelihood(self(_in)).mean
                xn.append(_xn)

            xn = torch.stack(xn, 0).view(horizon + 1, -1)
        return xn.cpu()

    def kstep_mse(self, xn, x, u, horizon):
        from sklearn.metrics import mean_squared_error, explained_variance_score

        mse, evar = [], []
        for _x, _u, _xn in zip(x, u, xn):
            _target, _prediction = [], []
            for t in range(_x.shape[0] - horizon + 1):
                _xn_hat = self.forcast(_x[t, :], _u[t:t + horizon, :], horizon)

                # -1 because xn is just x shifted by +1
                _target.append(_xn.numpy()[t + horizon - 1, :])
                _prediction.append(_xn_hat.numpy()[-1, :])

            _target = np.vstack(_target)
            _prediction = np.vstack(_prediction)

            _mse = mean_squared_error(_target, _prediction)
            mse.append(_mse)

            _evar = explained_variance_score(_target, _prediction,
                                             multioutput='variance_weighted')
            evar.append(_evar)

        return np.mean(mse), np.mean(evar)

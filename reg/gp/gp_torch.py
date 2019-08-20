import torch
from torch.optim import Adam

import gpytorch

from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood


class GPRegressor(gpytorch.models.ExactGP):

    def __init__(self, input, target):
        _likelihood = GaussianLikelihood()
        super(GPRegressor, self).__init__(input, target, _likelihood)

        self.input = input
        self.target = target

        self.mean_module = ZeroMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def fit(self, nb_iter=100):
        self.train()
        self.likelihood.train()

        optimizer = Adam([{'params': self.parameters()}], lr=1e-1)
        mll = ExactMarginalLogLikelihood(self.likelihood, self)

        for i in range(nb_iter):
            optimizer.zero_grad()
            _output = self(self.input)
            loss = - mll(_output, self.target)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, nb_iter, loss.item()))
            optimizer.step()

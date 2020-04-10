import torch
from torch.optim import Adam

import gpytorch

from gpytorch.means import ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood


class GPRegressor(gpytorch.models.ExactGP):

    def __init__(self, device='cpu'):
        if device == 'gpu' and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        _likelihood = GaussianLikelihood()
        super(GPRegressor, self).__init__(train_inputs=None,
                                          train_targets=None,
                                          likelihood=_likelihood)

        self.mean_module = ZeroMean()
        self.covar_module = ScaleKernel(RBFKernel())

    @property
    def model(self):
        return self

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def fit(self, input, target, nb_iter=100, lr=1e-1, verbose=True):
        input = input.to(self.device)
        target = target.to(self.device)

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

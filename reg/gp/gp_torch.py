import numpy as np

import torch
from torch.optim import Adam

import gpytorch

from gpytorch.means import ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood

from gpytorch.settings import max_preconditioner_size
from gpytorch.settings import max_root_decomposition_size
from gpytorch.settings import fast_pred_var

from sklearn.preprocessing import StandardScaler

from reg.gp.utils import transform, inverse_transform
from reg.gp.utils import ensure_args_torch_floats
from reg.gp.utils import ensure_res_numpy_floats
from reg.gp.utils import atleast_2d


class GPRegressor(gpytorch.models.ExactGP):

    def __init__(self, input_size, device='cpu'):
        if device == 'gpu' and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        self.input_size = input_size

        _likelihood = GaussianLikelihood()
        super(GPRegressor, self).__init__(train_inputs=None,
                                          train_targets=None,
                                          likelihood=_likelihood)

        self.mean_module = ZeroMean()
        self.covar_module = ScaleKernel(RBFKernel())

        self.input_trans = None
        self.target_trans = None

    @property
    def model(self):
        return self

    def forward(self, input):
        mean = self.mean_module(input)
        covar = self.covar_module(input)
        return MultivariateNormal(mean, covar)

    @ensure_args_torch_floats
    @ensure_res_numpy_floats
    def predict(self, input):
        self.model.eval()
        self.likelihood.eval()

        with max_preconditioner_size(10), torch.no_grad():
            with max_root_decomposition_size(30), fast_pred_var():
                input = transform(input, self.input_trans).to(self.device)
                input = atleast_2d(input, self.input_size)

                output = self.likelihood(self.model(input)).mean
                output = inverse_transform(output.cpu(), self.target_trans)

        return output

    def init_preprocess(self, target, input):
        self.target_trans = StandardScaler()
        self.input_trans = StandardScaler()

        self.target_trans.fit(target[:, np.newaxis])
        self.input_trans.fit(input.reshape(-1, self.input_size))

    @ensure_args_torch_floats
    def fit(self, target, input, nb_iter=100, lr=1e-1,
            verbose=True, preprocess=False):

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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

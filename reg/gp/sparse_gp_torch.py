import numpy as np

import torch
from torch.optim import SGD

import gpytorch

from gpytorch.means import ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood

from sklearn.decomposition import PCA

from reg.gp.utils import transform, inverse_transform
from reg.gp.utils import ensure_args_torch_floats
from reg.gp.utils import ensure_res_numpy_floats
from reg.gp.utils import atleast_2d


class SparseGPRegressor(gpytorch.models.ExactGP):

    @ensure_args_torch_floats
    def __init__(self, input, inducing_size, device='cpu'):
        if device == 'gpu' and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        if input.ndim == 1:
            self.input_size = 1
        else:
            self.input_size = input.shape[-1]

        _likelihood = GaussianLikelihood()
        super(SparseGPRegressor, self).__init__(train_inputs=None,
                                                train_targets=None,
                                                likelihood=_likelihood)

        self.mean_module = ZeroMean()
        self.base_covar_module = ScaleKernel(RBFKernel())

        inducing_idx = np.random.choice(len(input), inducing_size, replace=False)

        self.covar_module = InducingPointKernel(self.base_covar_module,
                                                inducing_points=input[inducing_idx, ...],
                                                likelihood=_likelihood)

        self.input_trans = None
        self.target_trans = None

    @property
    def model(self):
        return self

    def forward(self, input):
        mean = self.mean_module(input)
        covar = self.covar_module(input)
        output = MultivariateNormal(mean, covar)
        return output

    @ensure_args_torch_floats
    @ensure_res_numpy_floats
    def predict(self, input):
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            input = transform(input, self.input_trans)
            input = atleast_2d(input, self.input_size)

            output = self.likelihood(self.model(input)).mean
            output = inverse_transform(output.cpu(), self.target_trans)

        return output

    def init_preprocess(self, target, input):
        self.target_trans = PCA(n_components=1, whiten=True)
        self.input_trans = PCA(n_components=self.input_size, whiten=True)

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

        optimizer = SGD(self.model.parameters(), lr=lr)
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

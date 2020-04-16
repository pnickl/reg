import numpy as np
import torch
from torch.optim import SGD

import gpytorch

from gpytorch.means import MultitaskMean, ZeroMean
from gpytorch.kernels import MultitaskKernel, RBFKernel, InducingPointKernel
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import MultitaskGaussianLikelihood

from sklearn.decomposition import PCA

from reg.gp.utils import transform, inverse_transform
from reg.gp.utils import ensure_args_torch_floats
from reg.gp.utils import ensure_res_numpy_floats
from reg.gp.utils import atleast_2d


class SparseMultiGPRegressor(gpytorch.models.ExactGP):

    def __init__(self, target_size, input, inducing_size, device='cpu'):
        if device == 'gpu' and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        if input.ndim == 1:
            self.input_size = 1
        else:
            self.input_size = input.shape[-1]
        self.target_size = target_size

        _likelihood = MultitaskGaussianLikelihood(num_tasks=self.target_size)
        super(SparseMultiGPRegressor, self).__init__(train_inputs=None,
                                                     train_targets=None,
                                                     likelihood=_likelihood)

        self.mean_module = MultitaskMean(ZeroMean(), num_tasks=self.target_size)
        self.base_covar_module = MultitaskKernel(RBFKernel(), num_tasks=self.target_size, rank=1)

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
        output = MultitaskMultivariateNormal(mean, covar)
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
        self.target_trans = PCA(n_components=self.target_size, whiten=True)
        self.input_trans = PCA(n_components=self.input_size, whiten=True)

        self.target_trans.fit(target.reshape(-1, self.target_size))
        self.input_trans.fit(input.reshape(-1, self.input_size))

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

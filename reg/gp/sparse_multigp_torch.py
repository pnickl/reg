import numpy as np
import torch
from torch.optim import Adam

import gpytorch

from gpytorch.means import MultitaskMean, ZeroMean, ConstantMean
from gpytorch.kernels import MultitaskKernel, RBFKernel, ScaleKernel, InducingPointKernel
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.mlls import SumMarginalLogLikelihood

from gpytorch.settings import max_preconditioner_size
from gpytorch.settings import max_root_decomposition_size
from gpytorch.settings import fast_pred_var

from gpytorch.models import IndependentModelList
from gpytorch.likelihoods import LikelihoodList

from sklearn.preprocessing import StandardScaler

from reg.gp import SparseGPRegressor

from reg.gp.utils import transform, inverse_transform
from reg.gp.utils import ensure_args_torch_floats
from reg.gp.utils import ensure_res_numpy_floats
from reg.gp.utils import atleast_2d


class SparseMultiGPRegressor:

    @ensure_args_torch_floats
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

        self.inducing_size = inducing_size

        _list = [SparseGPRegressor(input, inducing_size)
                 for _ in range(self.target_size)]

        self.model = IndependentModelList(*[_model for _model in _list])
        self.likelihood = LikelihoodList(*[_model.likelihood for _model in _list])

        self.input_trans = None
        self.target_trans = None

    @ensure_args_torch_floats
    @ensure_res_numpy_floats
    def predict(self, input):
        self.model.eval()
        self.likelihood.eval()

        with max_preconditioner_size(25), torch.no_grad():
            with max_root_decomposition_size(30), fast_pred_var():
                input = transform(input, self.input_trans).to(self.device)
                input = atleast_2d(input, self.input_size)

                _input = [input for _ in range(self.target_size)]
                predictions = self.likelihood(*self.model(*_input))
                output = torch.stack([_pred.mean.cpu() for _pred in predictions]).T
                output = inverse_transform(output, self.target_trans)

        return output

    def init_preprocess(self, target, input):
        self.target_trans = StandardScaler()
        self.input_trans = StandardScaler()

        self.target_trans.fit(target.reshape(-1, self.target_size))
        self.input_trans.fit(input.reshape(-1, self.input_size))

    @ensure_args_torch_floats
    def fit(self, target, input, nb_iter=100, lr=1e-1,
            verbose=True, preprocess=True):

        if preprocess:
            self.init_preprocess(target, input)
            target = transform(target, self.target_trans)
            input = transform(input, self.input_trans)

            # update inducing points
            inducing_idx = np.random.choice(len(input), self.inducing_size, replace=False)
            for i, _model in enumerate(self.model.models):
                _model.covar_module.inducing_points.data = input[inducing_idx, ...]

        target = target.to(self.device)
        input = input.to(self.device)

        for i, _model in enumerate(self.model.models):
            _model.set_train_data(input, target[:, i], strict=False)

        self.model.train().to(self.device)
        self.likelihood.train().to(self.device)

        optimizer = Adam([{'params': self.model.parameters()}], lr=lr)
        mll = SumMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(nb_iter):
            optimizer.zero_grad()
            _output = self.model(*self.model.train_inputs)
            loss = - mll(_output, self.model.train_targets)
            loss.backward()
            if verbose:
                print('Iter %d/%d - Loss: %.3f' % (i + 1, nb_iter, loss.item()))
            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

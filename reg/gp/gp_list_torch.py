import numpy as np
import torch
from torch.optim import Adam

import gpytorch

from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import MultitaskGaussianLikelihood

from gpytorch.settings import max_preconditioner_size
from gpytorch.settings import max_root_decomposition_size
from gpytorch.settings import fast_pred_var

from sklearn.preprocessing import StandardScaler

from reg.gp.utils import transform, inverse_transform
from reg.gp.utils import ensure_args_torch_floats
from reg.gp.utils import ensure_res_numpy_floats
from reg.gp.utils import atleast_2d


class GPListRegressor(gpytorch.models.ExactGP):

    def __init__(self, input_size, target_size, device='cpu'):
        if device == 'gpu' and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        self.input_size = input_size
        self.target_size = target_size

        _likelihood = MultitaskGaussianLikelihood(num_tasks=self.target_size)
        super(GPListRegressor, self).__init__(train_inputs=None,
                                              train_targets=None,
                                              likelihood=_likelihood)

        self.mean_module = ConstantMean(batch_shape=torch.Size([self.target_size]))
        self.covar_module = ScaleKernel(RBFKernel(batch_shape=torch.Size([self.target_size])),
                                        batch_shape=torch.Size([self.target_size]))

        self.input_trans = None
        self.target_trans = None

    @property
    def model(self):
        return self

    def forward(self, input):
        mean = self.mean_module(input)
        covar = self.covar_module(input)
        return MultitaskMultivariateNormal.from_batch_mvn(MultivariateNormal(mean, covar))

    @ensure_args_torch_floats
    @ensure_res_numpy_floats
    def predict(self, input):
        self.device = torch.device('cpu')

        self.model.eval().to(self.device)
        self.likelihood.eval().to(self.device)

        with max_preconditioner_size(10), torch.no_grad():
            with max_root_decomposition_size(30), fast_pred_var():
                input = transform(input, self.input_trans)
                input = atleast_2d(input, self.input_size)

                output = self.likelihood(self.model(input)).mean
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


class DynamicListGPRegressor(GPListRegressor):
    def __init__(self, input_size, target_size, incremental=True, device='cpu'):
        super(DynamicListGPRegressor, self).__init__(input_size, target_size, device)

        self.incremental = incremental

    @ensure_args_torch_floats
    @ensure_res_numpy_floats
    def predict(self, input):
        with max_preconditioner_size(10), torch.no_grad():
            with max_root_decomposition_size(30), fast_pred_var():
                _input = transform(input, self.input_trans).to(self.device)
                _input = atleast_2d(_input, self.input_size)

                output = self.likelihood(self.model(_input)).mean
                output = inverse_transform(output, self.target_trans)
                output = output.reshape((self.target_size, ))

        if self.incremental:
            return input[..., :self.target_size] + output
        else:
            return output

    def forcast(self, state, exogenous, horizon=1):
        self.device = torch.device('cpu')

        self.model.eval().to(self.device)
        self.likelihood.eval().to(self.device)

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

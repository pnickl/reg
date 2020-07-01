import torch
import torch.nn as nn

from torch.nn.modules.loss import MSELoss
from torch.optim import Adam

from torch.utils.data import BatchSampler, SubsetRandomSampler

import numpy as np
import numpy.random as npr
from numpy.random import multivariate_normal as mvn

from sklearn.decomposition import PCA

from reg.rbf.torch.utils import to_float
from reg.rbf.torch.utils import transform, inverse_transform
from reg.rbf.torch.utils import ensure_args_torch_floats
from reg.rbf.torch.utils import ensure_res_numpy_floats
from reg.rbf.torch.utils import ensure_args_atleast_2d


class FourierFeatures:

    def __init__(self, sizes, bandwidth):

        self.sizes = sizes
        self.input_size = self.sizes[0]
        self.hidden_size = self.sizes[1]

        self.freq = to_float(mvn(mean=np.zeros((self.input_size, )),
                                 cov=np.diag(1.0 / bandwidth),
                                 size=self.hidden_size))

        self.shift = to_float(npr.uniform(-np.pi, np.pi, size=self.hidden_size))

    def fit_transform(self, x):
        return torch.sin(torch.einsum('nk,...k->...n', self.freq, x) + self.shift)


class FourierRegressor(nn.Module):

    def __init__(self, sizes, bandwidth):
        super(FourierRegressor, self).__init__()

        self.bandwidth = bandwidth

        self.sizes = sizes

        self.target_size = self.sizes[-1]
        self.input_size = self.sizes[0]
        self.hidden_size = self.sizes[1]

        self.basis = FourierFeatures(self.sizes, self.bandwidth)
        self.output = nn.Linear(self.hidden_size, self.target_size)

        self.criterion = MSELoss()
        self.optim = None

        self.input_trans = None
        self.target_trans = None

    @torch.no_grad()
    def features(self, input):
        return self.basis.fit_transform(input)

    @ensure_args_torch_floats
    def forward(self, input):
        return self.output(self.features(input))

    @ensure_args_torch_floats
    @ensure_res_numpy_floats
    def predict(self, input):
        input = input.reshape((-1, self.input_size))
        input = transform(input, self.input_trans)

        with torch.no_grad():
            output = self.forward(input)

        output = inverse_transform(output, self.target_trans)
        return torch.squeeze(output)

    def init_preprocess(self, target, input):
        self.target_trans = PCA(n_components=self.target_size, whiten=True)
        self.input_trans = PCA(n_components=self.input_size, whiten=True)

        self.target_trans.fit(target)
        self.input_trans.fit(input)

    @ensure_args_torch_floats
    @ensure_args_atleast_2d
    def fit(self, target, input, nb_epochs=1000, batch_size=32,
            lr=1e-3, l2=1e-32, verbose=True, preprocess=True):

        if preprocess:
            self.init_preprocess(target, input)
            target = transform(target, self.target_trans)
            input = transform(input, self.input_trans)

        self.optim = Adam(self.parameters(), lr=lr, weight_decay=l2)

        set_size = input.shape[0]
        batch_size = set_size if batch_size is None else batch_size

        for n in range(nb_epochs):
            batches = list(BatchSampler(SubsetRandomSampler(range(set_size)), batch_size, True))

            for batch in batches:
                self.optim.zero_grad()
                _output = self.forward(input[batch])
                loss = self.criterion(_output, target[batch])
                loss.backward()
                self.optim.step()

            if verbose:
                if n % 50 == 0:
                    output = self.forward(input)
                    print('Epoch: {}/{}.............'.format(n, nb_epochs), end=' ')
                    print("Loss: {:.4f}".format(self.criterion(output, target)))

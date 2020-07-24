import numpy as np
import numpy.random as npr
from numpy.random import multivariate_normal as mvn

from mimo.distributions import MatrixNormalWishart
from mimo.distributions import LinearGaussianWithMatrixNormalWishart

from sklearn.preprocessing import StandardScaler

from reg.rbf.npy.utils import transform, inverse_transform
from reg.rbf.npy.utils import ensure_args_atleast_2d


class FourierFeatures:

    def __init__(self, sizes, bandwidth):

        self.sizes = sizes
        self.input_size = self.sizes[0]
        self.hidden_size = self.sizes[1]

        self.freq = mvn(mean=np.zeros((self.input_size, )),
                        cov=np.diag(1.0 / bandwidth),
                        size=self.hidden_size)

        self.shift = npr.uniform(-np.pi, np.pi, size=self.hidden_size)

    def fit_transform(self, x):
        return np.sin(np.einsum('nk,...k->...n', self.freq, x) + self.shift)


class BayesianFourierRegressor:

    def __init__(self, sizes, bandwidth, prior=None):
        super(BayesianFourierRegressor, self).__init__()

        self.bandwidth = bandwidth

        self.sizes = sizes

        self.target_size = self.sizes[-1]
        self.input_size = self.sizes[0]
        self.hidden_size = self.sizes[1]

        self.basis = FourierFeatures(self.sizes, self.bandwidth)

        if prior is None:
            # A standard relatively uninformative prior
            hypparams = dict(M=np.zeros((self.target_size, self.hidden_size + 1)),
                             K=1e-2 * np.eye(self.hidden_size + 1),
                             psi=np.eye(self.target_size),
                             nu=self.target_size + 1)
            prior = MatrixNormalWishart(**hypparams)

        self.model = LinearGaussianWithMatrixNormalWishart(prior, affine=True)

        self.input_trans = None
        self.target_trans = None

    def features(self, input):
        return self.basis.fit_transform(input)

    def predict(self, input):
        input = transform(input.reshape((-1, self.input_size)), self.input_trans)

        feat = self.features(input)
        output, _ = self.model.posterior_predictive_gaussian(np.squeeze(feat))

        output = inverse_transform(output, self.target_trans).squeeze()
        return output

    def init_preprocess(self, target, input):
        self.target_trans = StandardScaler()
        self.input_trans = StandardScaler()

        self.target_trans.fit(target)
        self.input_trans.fit(input)

    @ensure_args_atleast_2d
    def fit(self, target, input, preprocess=True, nb_iter=3):

        if preprocess:
            self.init_preprocess(target, input)
            target = transform(target, self.target_trans)
            input = transform(input, self.input_trans)

        feat = self.features(input)
        for _ in range(nb_iter):
            # do empirical bayes
            self.model.meanfield_update(y=target, x=feat)
            self.model.prior = self.model.posterior

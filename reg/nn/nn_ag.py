import autograd.numpy as np
import autograd.numpy.random as npr

from autograd import grad
from autograd.misc.optimizers import adam

from reg.nn.utils import logistic, relu, linear, tanh
from reg.nn.utils import mse, ce


class NNRegressor:

    def __init__(self, sizes, nonlin='tanh',
                 output='logistic', loss='ce'):

        self.nb_layers = len(sizes) - 2
        self.sizes = sizes

        nlist = dict(relu=relu, tanh=tanh, logistic=logistic, linear=linear)
        self.nonlins = [nlist[nonlin]] * self.nb_layers + [nlist[output]]

        self.params = [(npr.randn(x, y), npr.randn(y))
                       for i, (x, y) in enumerate(zip(self.sizes[:-1], self.sizes[1:]))]

        llist = dict(mse=mse, ce=ce)
        self.criterion = llist[loss]

    def forward(self, x):
        _out = x
        for i, (w, b) in enumerate(self.params):
            nonlin = self.nonlins[i]
            act = np.einsum('nk,kh->nh', _out, w) + b
            _out = nonlin(act)[0]
        return _out

    def fit(self, y, x, nb_epochs=500, batch_size=16, lr=1e-3):

        nb_batches = int(np.ceil(len(x) / batch_size))

        def batch_indices(iter):
            idx = iter % nb_batches
            return slice(idx * batch_size, (idx + 1) * batch_size)

        def _objective(params, iter):
            self.params = params
            idx = batch_indices(iter)
            return self.cost(y[idx], x[idx])

        def _callback(params, iter, grad):
            if iter % (nb_batches * 10) == 0:
                self.params = params
                print('Epoch: {}/{}.............'.format(iter // nb_batches, nb_epochs), end=' ')
                print("Loss: {:.4f}".format(self.cost(y, x)))

        _gradient = grad(_objective)

        self.params = adam(_gradient, self.params, step_size=lr,
            num_iters=nb_epochs * nb_batches, callback=_callback)

    def cost(self, y, x):
        _y = self.forward(x)
        return self.criterion(y, _y)[0]

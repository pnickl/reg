import autograd.numpy as np
import autograd.numpy.random as npr

from autograd import grad
from autograd.misc.optimizers import sgd, adam

from reg.nn.utils import logistic, relu, linear, tanh
from reg.nn.utils import mse, ce


class Layer:

    def __init__(self, size, nonlin='tanh'):
        self.size = size

        self.weights = npr.randn(self.size[0], self.size[-1])
        self.biases = npr.randn(self.size[-1])

        self._in = None
        self._out = None
        self._act = None

        nlist = dict(relu=relu, tanh=tanh, logistic=logistic, linear=linear)
        self.nonlin = nlist[nonlin]

    def forward(self, x):
        self._in = x
        self._act = np.einsum('nk,kh->nh', self._in, self.weights) + self.biases
        self._out = self.nonlin(self._act)[0]
        return self._out

    def backward(self, dh):
        # cost gradient w.r.t. act dc/dact
        da = dh * self.nonlin(self._act)[1]

        # cost gradient w.r.t. weights dc/dw
        dw = np.einsum('nk,nh->nkh', self._in, da)

        # cost gradient w.r.t. biases dc/db
        db = da

        # cost gradient w.r.t. input dc/dx
        dx = np.einsum('kh,nh->nk', self.weights, da)

        return dx, dw, db


class NNRegressor:

    def __init__(self, sizes, nonlin='tanh',
                 output='logistic', loss='ce'):

        self.nb_layers = len(sizes) - 2
        self.sizes = sizes

        nlist = [nonlin] * self.nb_layers + [output]
        self.layers = [Layer([x, y], nlist[i])
                       for i, (x, y) in enumerate(zip(self.sizes[:-1], self.sizes[1:]))]

        llist = dict(mse=mse, ce=ce)
        self.criterion = llist[loss]

    def forward(self, x):
        out = x
        for l in self.layers:
            out = l.forward(out)
        return out

    @property
    def params(self):
        return [(l.weights, l.biases) for l in self.layers]

    @params.setter
    def params(self, values):
        for i, (w, b) in enumerate(values):
            self.layers[i].weights = w
            self.layers[i].biases = b

    def backprop(self, y, x):
        dW, dB = [], []

        # cost gradient w.r.t. output dc/dout
        _, dx = self.cost(y, x)

        for l in reversed(self.layers):
            dx, dw, db = l.backward(dx)

            dW.insert(0, np.sum(dw, axis=0))
            dB.insert(0, np.sum(db, axis=0))

        return dW, dB

    def fit(self, y, x, nb_epochs=500, batch_size=16, lr=1e-3):

        nb_batches = int(np.ceil(len(x) / batch_size))

        def batch_indices(iter):
            idx = iter % nb_batches
            return slice(idx * batch_size, (idx + 1) * batch_size)

        def _gradient(params, iter):
            self.params = params
            idx = batch_indices(iter)
            return self.backprop(y[idx], x[idx])

        def _callback(params, iter, grad):
            if iter % (nb_batches * 10) == 0:
                self.params = params
                print('Epoch: {}/{}.............'.format(iter // nb_batches, nb_epochs), end=' ')
                print("Loss: {:.4f}".format(self.cost(y, x)[0]))

        self.params = adam(_gradient, self.params, step_size=lr,
                           num_iters=nb_epochs * nb_batches, callback=_callback)

    def cost(self, y, x):
        _y = self.forward(x)
        return self.criterion(y, _y)

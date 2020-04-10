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

        self._input = None
        self._output = None
        self._activation = None

        nlist = dict(relu=relu, tanh=tanh, logistic=logistic, linear=linear)
        self.nonlin = nlist[nonlin]

    def forward(self, input):
        self._input = input
        self._activation = np.einsum('nk,kh->nh', self._input, self.weights) + self.biases
        self._output = self.nonlin(self._activation)[0]
        return self._output

    def backward(self, dh):
        # cost gradient w.r.t. act dcost/dactivation
        da = dh * self.nonlin(self._activation)[1]

        # cost gradient w.r.t. weights dcost/dweights
        dw = np.einsum('nk,nh->nkh', self._input, da)

        # cost gradient w.r.t. biases dcost/dbias
        db = da

        # cost gradient w.r.t. input dcost/dinput
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

    def forward(self, input):
        output = input
        for l in self.layers:
            output = l.forward(output)
        return output

    @property
    def params(self):
        return [(l.weights, l.biases) for l in self.layers]

    @params.setter
    def params(self, values):
        for i, (w, b) in enumerate(values):
            self.layers[i].weights = w
            self.layers[i].biases = b

    def backprop(self, target, input):
        dW, dB = [], []

        # cost gradient w.r.t. output dcost/doutput
        _, dx = self.cost(target, input)

        for l in reversed(self.layers):
            dx, dw, db = l.backward(dx)

            dW.insert(0, np.sum(dw, axis=0))
            dB.insert(0, np.sum(db, axis=0))

        return dW, dB

    def fit(self, target, input, nb_epochs=500, batch_size=16, lr=1e-3, verbose=True):

        nb_batches = int(np.ceil(len(input) / batch_size))

        def batch_indices(iter):
            idx = iter % nb_batches
            return slice(idx * batch_size, (idx + 1) * batch_size)

        def _gradient(params, iter):
            self.params = params
            idx = batch_indices(iter)
            return self.backprop(target[idx], input[idx])

        def _callback(params, iter, grad):
            if iter % (nb_batches * 10) == 0:
                self.params = params
                if verbose:
                    print('Epoch: {}/{}.............'.format(iter // nb_batches, nb_epochs), end=' ')
                    print("Loss: {:.4f}".format(self.cost(target, input)[0]))

        self.params = adam(_gradient, self.params, step_size=lr,
                           num_iters=nb_epochs * nb_batches, callback=_callback)

    def cost(self, target, input):
        _output = self.forward(input)
        return self.criterion(target, _output)

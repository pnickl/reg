import autograd.numpy as np
import autograd.numpy.random as npr

from autograd import grad
from autograd.misc.optimizers import sgd, adam

from nn_util import logistic, relu, linear, tanh
from nn_util import mse, ce


class Layer():

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
        self._act = np.einsum('nk,kh->nh', x, self.weights) + self.biases
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


class Network():

    def __init__(self, sizes, nonlin='tanh',
                 output='logistic', loss='ce'):

        self.nb_layers = len(sizes) - 2
        self.sizes = sizes

        nlist = [nonlin] * self.nb_layers + [output]
        self.layers = [Layer([x, y], nlist[i])
                       for i, (x, y) in enumerate(zip(self.sizes[:-1], self.sizes[1:]))]

        llist = dict(mse=mse, ce=ce)
        self.loss = llist[loss]

    def forward(self, x):
        _out = x
        for l in self.layers:
            _out = l.forward(_out)

        return _out

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
            if iter % nb_batches == 0:
                self.params = params
                print('iter=', iter, 'cost=', self.cost(y, x)[0])

        self.params = adam(_gradient, self.params, step_size=lr,
            num_iters=nb_epochs * nb_batches, callback=_callback)

    def cost(self, y, x):
        _y = self.forward(x)
        return self.loss(y, _y)


if __name__ == '__main__':

    # npr.seed(1337)

    from sklearn.datasets import load_breast_cancer, load_digits, load_boston

    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder

    # regression example
    set = load_boston()

    x, y = set['data'], set['target']
    y = y[:, np.newaxis]

    xt, xv, yt, yv = train_test_split(x, y, test_size=0.2)

    nb_in = x.shape[-1]
    nb_out = y.shape[-1]

    nn = Network([nb_in, 10, nb_out], nonlin='relu', output='linear', loss='mse')
    nn.fit(yt, xt, nb_epochs=250, batch_size=64, lr=1e-2)

    print('train:', 'cost=', nn.cost(yt, xt)[0])
    print('test:', 'cost=', nn.cost(yv, xv)[0])


    # # classification example
    # set = load_breast_cancer()
    # # set = load_digits()
    #
    # x, y = set['data'], set['target']
    #
    # enc = OneHotEncoder(categories='auto')
    # y = enc.fit_transform(y[:, np.newaxis]).toarray()
    #
    # xt, xv, yt, yv = train_test_split(x, y, test_size=0.2)
    #
    # scaler = StandardScaler()
    # xt = scaler.fit_transform(xt)
    # xv = scaler.fit_transform(xv)
    #
    # nb_in = x.shape[-1]
    # nb_out = y.shape[-1]
    #
    # nn = Network([nb_in, 4, nb_out], nonlin='tanh', output='logistic', loss='ce')
    # nn.fit(yt, xt, nb_epochs=250, batch_size=64, lr=1e-2)
    #
    # _yt = nn.forward(xt)
    # class_error = np.linalg.norm(yt - np.rint(_yt), axis=1)
    # print('train:', 'cost=', nn.cost(yt, xt)[0], 'class. error=', np.mean(class_error))
    #
    # _yv = nn.forward(xv)
    # class_error = np.linalg.norm(yv - np.rint(_yv), axis=1)
    # print('test:', 'cost=', nn.cost(yv, xv)[0], 'class. error=', np.mean(class_error))

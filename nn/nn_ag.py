import autograd.numpy as np
import autograd.numpy.random as npr

from autograd import grad
from autograd.misc.optimizers import sgd, adam

from reg.nn.util import logistic, relu, linear, tanh
from reg.nn.util import mse, ce


class Network:

    def __init__(self, sizes, nonlin='tanh',
                 output='logistic', loss='ce'):

        self.nb_layers = len(sizes) - 2
        self.sizes = sizes

        nlist = dict(relu=relu, tanh=tanh, logistic=logistic, linear=linear)
        self.nonlins = [nlist[nonlin]] * self.nb_layers + [nlist[output]]

        self.params = [(npr.randn(x, y), npr.randn(y))
                       for i, (x, y) in enumerate(zip(self.sizes[:-1], self.sizes[1:]))]

        llist = dict(mse=mse, ce=ce)
        self.loss = llist[loss]

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
            if iter % nb_batches == 0:
                self.params = params
                print('iter=', iter, 'cost=', self.cost(y, x))

        _gradient = grad(_objective)

        self.params = adam(_gradient, self.params, step_size=lr,
            num_iters=nb_epochs * nb_batches, callback=_callback)

    def cost(self, y, x):
        _y = self.forward(x)
        return self.loss(y, _y)[0]


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

    print('train:', 'cost=', nn.cost(yt, xt))
    print('test:', 'cost=', nn.cost(yv, xv))

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
    # print('train:', 'cost=', nn.cost(yt, xt), 'class. error=', np.mean(class_error))
    #
    # _yv = nn.forward(xv)
    # class_error = np.linalg.norm(yv - np.rint(_yv), axis=1)
    # print('test:', 'cost=', nn.cost(yv, xv), 'class. error=', np.mean(class_error))

import autograd.numpy as np
import autograd.numpy.random as npr


class Perceptron:

    def __init__(self, input_size):
        self.input_size = input_size

        self.weights = npr.randn(self.input_size)
        self.bias = npr.randn()

    def forward(self, x):
        a = np.einsum('k,...k->...', self.weights, x) + self.bias
        return np.clip(np.sign(a), 0.0, 1.0)

    def fit(self, y, x, nb_epochs, lr=1e-3):
        nb_samples = y.shape[0]

        for l in range(nb_epochs):
            for n in range(nb_samples):
                self.weights = self.weights - lr * (self.forward(x[n, :]) - y[n]) * x[n, :]
                self.bias = self.bias - lr * (self.forward(x[n, :]) - y[n])

            if l % 10 == 0:
                print('Epoch: {}/{}.............'.format(l, nb_epochs), end=' ')
                print("Loss: {:.4f}".format(self.error(y, x)))

    def error(self, y, x):
        _y = self.forward(x)
        return np.mean(y != _y, axis=0)

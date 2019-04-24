import autograd.numpy as np
import autograd.numpy.random as npr


class Perceptron:

    def __init__(self, nb_in):
        self.nb_in = nb_in

        self.weights = npr.randn(self.nb_in)
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

            print("epoch=", l,  "class. error=", self.error(y, x))

    def error(self, y, x):
        _y = perc.forward(x)
        return np.mean(y != _y, axis=0)


if __name__ == '__main__':

    npr.seed(1337)

    from sklearn.datasets import load_breast_cancer
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    set = load_breast_cancer()
    x, y = set['data'], set['target']

    xt, xv, yt, yv = train_test_split(x, y, test_size=0.2)

    scaler = StandardScaler()
    xt = scaler.fit_transform(xt)
    xv = scaler.fit_transform(xv)

    nb_in = x.shape[-1]

    perc = Perceptron(nb_in)
    perc.fit(yt, xt, nb_epochs=250, lr=0.25)

    print("testing", "class. error=", perc.error(yv, xv))

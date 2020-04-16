import numpy as np
import matplotlib.pyplot as plt

from reg.nn.nn_torch import NNRegressor


if __name__ == "__main__":

    # data
    input = np.linspace(0, 1, 25)
    target = np.sin(input * (2 * np.pi)) + np.random.randn(input.shape[0]) * 0.1

    # build model
    model = NNRegressor([1, 8, 8, 1], nonlin='tanh')
    model.fit(target, input, nb_epochs=2500, preprocess=True)

    output = model.predict(input)

    plt.scatter(input, target, s=5, color='r', zorder=10)
    plt.plot(input, output, '-o', color='b', zorder=1)
    plt.show()

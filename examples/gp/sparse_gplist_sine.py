import numpy as np
import numpy.random as npr

import matplotlib.pyplot as plt

from reg.gp import SparseGPListRegressor


if __name__ == "__main__":

    np.random.seed(1337)

    # data
    x1 = np.linspace(0, 1, 50)
    y1 = 1.5 * np.sin(x1 * (2 * np.pi)) + npr.randn(x1.shape[0]) * 0.1

    x2 = 2. * np.linspace(0, 1, 50)
    y2 = 2. * np.sin(x2 * 2 * (2 * np.pi)) + npr.randn(x2.shape[0]) * 0.1

    input = np.vstack((x1, x2)).T
    target = np.vstack((y1, y2)).T

    # build model
    model = SparseGPListRegressor(2, input, 15)
    model.fit(target, input, preprocess=True)

    output = model.predict(input)

    for i in range(2):
        plt.figure()
        plt.scatter(input[:, i], target[:, i], s=5, color='r', zorder=10)
        plt.plot(input[:, i], output[:, i], '-o', color='b', zorder=1)
    plt.show()

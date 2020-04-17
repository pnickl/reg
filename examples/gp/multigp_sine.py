import numpy as np
import matplotlib.pyplot as plt

from reg.gp import MultiGPRegressor


if __name__ == "__main__":

    np.random.seed(1337)

    # data
    x1 = np.linspace(0, 1, 50)
    y1 = np.sin(x1 * (2 * np.pi)) + np.random.randn(x1.shape[0]) * 0.1

    x2 = 2 * np.linspace(0, 1, 50)
    y2 = np.sin(x2 * 2 * (2 * np.pi)) + np.random.randn(x2.shape[0]) * 0.25

    input = np.vstack((x1, x2)).T
    target = np.vstack((y1, y2)).T

    # build model
    model = MultiGPRegressor(input_size=2, target_size=2)
    model.fit(target, input, preprocess=True)

    output = model.predict(input)

    for i in range(2):
        plt.figure()
        plt.scatter(input[:, i], target[:, i], s=5, color='r', zorder=10)
        plt.plot(input[:, i], output[:, i], '-o', color='b', zorder=1)
    plt.show()
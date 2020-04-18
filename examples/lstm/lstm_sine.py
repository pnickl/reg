import torch
import numpy as np

from reg.nn import LSTMRegressor


if __name__ == '__main__':

    np.random.seed(1337)
    torch.manual_seed(1337)

    import matplotlib.pyplot as plt

    T, L, N = 20, 250, 25

    input_size = 1
    target_size = 1

    x = np.empty((N, L), 'int64')
    x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
    data = np.sin(x / 1.0 / T).astype('float64')

    input = data[:, :-1].reshape(N, -1, input_size)
    target = data[:, 1:].reshape(N, -1, target_size)

    lstm = LSTMRegressor(input_size=input_size,
                         target_size=target_size,
                         hidden_size=24,
                         nb_layers=2)

    lstm.fit(target, input, nb_epochs=100, preprocess=True)

    horizon, buffer = 200, 35
    yhat = lstm.forcast(input[:, :buffer, :], horizon=horizon)

    plt.figure()
    plt.plot(target[5, buffer:buffer + horizon + 1, :], label='target')
    plt.plot(yhat[5, ...], label='prediction')
    plt.legend()
    plt.show()

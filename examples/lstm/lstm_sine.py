import torch
import numpy as np

from reg.nn import LSTMRegressor

to_double = lambda arr: torch.from_numpy(arr).double()


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    T, L, N = 20, 250, 10

    input_size = 1
    target_size = 1

    x = np.empty((N, L), 'int64')
    x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
    data = np.sin(x / 1.0 / T).astype('float64')

    input = to_double(data[:N - 1, :-1]).view(N - 1, -1, input_size)
    target = to_double(data[:N - 1, 1:]).view(N - 1, -1, target_size)

    test_input = to_double(data[N - 1, :-1]).view(-1, input_size)
    test_target = to_double(data[N - 1, 1:]).view(-1, target_size)

    lstm = LSTMRegressor(input_size,
                         target_size,
                         nb_neurons=[10, 10])

    lstm.fit(target, input, nb_epochs=25)

    horizon, buffer = 200, 10
    yhat = lstm.forcast(test_input[:buffer, :], horizon)

    plt.plot(test_target.numpy()[buffer:buffer + horizon + 1, :], label='target')
    plt.plot(yhat.numpy(), label='prediction')
    plt.legend()
    plt.show()

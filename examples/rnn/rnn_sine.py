import torch
import autograd.numpy as np

from reg.nn import RNNRegressor

to_float = lambda arr: torch.from_numpy(arr).float()


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    T, L, N = 20, 250, 10

    input_size = 1
    target_size = 1

    x = np.empty((N, L), 'int64')
    x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
    data = np.sin(x / 1.0 / T).astype('float64')

    input = to_float(data[:N - 1, :-1]).view(N - 1, -1, input_size)
    target = to_float(data[:N - 1, 1:]).view(N - 1, -1, target_size)

    test_input = to_float(data[N - 1, :-1]).view(-1, input_size)
    test_target = to_float(data[N - 1, 1:]).view(-1, target_size)

    rnn = RNNRegressor(input_size=input_size,
                       target_size=target_size,
                       hidden_size=25,
                       nb_layers=1)

    rnn.fit(target, input, nb_epochs=1000, lr=1.e-3)

    horizon, buffer = 200, 10
    yhat = rnn.forcast(test_input[:buffer, :], horizon)

    plt.plot(test_target.numpy()[buffer:buffer + horizon + 1, :], label='target')
    plt.plot(yhat.numpy(), label='prediction')
    plt.legend()
    plt.show()

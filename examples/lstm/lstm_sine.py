import torch
import torch.nn as nn
import numpy as np

from reg.nn import LSTMRegressor

to_torch = lambda arr: torch.from_numpy(arr).double()


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    T, L, N = 20, 250, 10

    input_size = 1
    output_size = 1

    x = np.empty((N, L), 'int64')
    x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
    data = np.sin(x / 1.0 / T).astype('float64')

    input = to_torch(data[3:, :-1])
    target = to_torch(data[3:, 1:])

    test_input = to_torch(data[:3, :-1])
    test_target = to_torch(data[:3, 1:])

    lstm = LSTMRegressor(input_size, output_size, nb_neurons=[10, 10])
    lstm.fit(target, input, 25)

    with torch.no_grad():
        future = 1000
        pred = lstm(test_input, future=future)
        y = pred.detach().numpy()

    plt.figure(figsize=(30, 10))
    plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    def draw(yi, color):
        plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth=2.0)
        plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth=2.0)

    draw(y[0], 'r')
    draw(y[1], 'g')
    draw(y[2], 'b')
    plt.show()
    plt.close()
